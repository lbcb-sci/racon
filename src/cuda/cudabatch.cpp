/*!
 * @file cudabatch.cpp
 *
 * @brief CUDABatch class source file
 */
#include "cudabatch.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>

#include "spoa/spoa.hpp"
#include "claraparabricks/genomeworks/utils/cudautils.hpp"

#include "cudautils.hpp"

namespace racon {

using namespace claraparabricks::genomeworks::cudapoa;  // NOLINT

std::atomic<std::uint32_t> CUDABatchProcessor::batches;

std::unique_ptr<CUDABatchProcessor> CreateCUDABatch(
    std::uint32_t max_window_depth,
    std::uint32_t device,
    std::size_t avail_mem,
    std::int8_t gap,
    std::int8_t mismatch,
    std::int8_t match,
    bool cuda_banded_alignment) {
  return std::unique_ptr<CUDABatchProcessor>(new CUDABatchProcessor(
      max_window_depth,
      device,
      avail_mem,
      gap,
      mismatch,
      match,
      cuda_banded_alignment));
}

CUDABatchProcessor::CUDABatchProcessor(
    std::uint32_t max_window_depth,
    std::uint32_t device,
    std::size_t avail_mem,
    std::int8_t gap,
    std::int8_t mismatch,
    std::int8_t match,
    bool cuda_banded_alignment)
    : windows_(),
      seqs_added_per_window_() {
  bid_ = CUDABatchProcessor::batches++;

  // Create new CUDA stream.
  GW_CU_CHECK_ERR(cudaStreamCreate(&stream_));

  BatchConfig batch_config(
      1023,
      max_window_depth,
      256,
      cuda_banded_alignment ? BandMode::static_band : BandMode::full_band);

  cudapoa_batch_ = create_batch(
      device,
      stream_,
      avail_mem,
      OutputType::consensus,
      batch_config,
      gap,
      mismatch,
      match);
}

CUDABatchProcessor::~CUDABatchProcessor() {
  // Destroy CUDA stream.
  GW_CU_CHECK_ERR(cudaStreamDestroy(stream_));
}

bool CUDABatchProcessor::AddWindow(
    std::shared_ptr<Window> window,
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences) {
  Group poa_group;

  std::uint32_t num_seqs = window->sequences_ids.size();
  std::vector<std::string> seqs;
  std::vector<std::vector<std::int8_t>> weights(num_seqs);

  for (std::uint32_t i = 0; i < num_seqs; ++i) {
    seqs.emplace_back(sequences[window->sequences_ids[i]]->InflateData(
        window->sequences_intervals[i].first,
        window->sequences_intervals[i].second - window->sequences_intervals[i].first));  // NOLINT

    if (sequences[window->sequences_ids[i]]->block_quality.empty()) {
      continue;
    }
    weights[i].reserve(seqs.back().size());
    std::uint32_t j = window->sequences_intervals[i].first;
    for (; j < window->sequences_intervals[i].second; ++j) {
      weights[i].emplace_back(sequences[window->sequences_ids[i]]->Score(j));
    }
  }

  // Add first sequence as backbone to graph.
  std::vector<std::int8_t> backbone_weights(window->backbone.size(), 0);
  Entry e = {
      window->backbone.c_str(),
      backbone_weights.data(),
      static_cast<std::int32_t>(window->backbone.size())
  };
  poa_group.push_back(e);

  // Add the rest of the sequences in sorted order of starting positions.
  std::vector<std::uint32_t> rank(num_seqs);
  std::iota(rank.begin(), rank.end(), 0);
  std::stable_sort(
      rank.begin(),
      rank.end(),
      [&] (std::uint32_t lhs, std::uint32_t rhs) {
        return window->positions[lhs].first < window->positions[rhs].first;
      });

  // Start from index 1 since first sequence has already been added as backbone.
  std::uint32_t long_seq = 0;
  std::uint32_t skipped_seq = 0;
  for (std::uint32_t j = 0; j < num_seqs; j++) {
    std::uint32_t i = rank[j];

    Entry p = {
        seqs[i].c_str(),
        weights[i].data(),
        static_cast<std::int32_t>(seqs[i].size())
    };

    poa_group.push_back(p);
  }

  // Add group to CUDAPOA batch object.
  std::vector<StatusType> entry_status;
  StatusType status = cudapoa_batch_->add_poa_group(entry_status, poa_group);

  weights.clear();
  seqs.clear();

  // If group was added, then push window in accepted windows list.
  if (status != StatusType::success) {
    return false;
  } else {
    windows_.push_back(window);
  }

  // Keep track of how many sequences were actually processed for this
  // group. This acts as the effective coverage for that window.
  std::int32_t seq_added = 0;
  for (std::uint32_t i = 1; i < entry_status.size(); i++) {
    if (entry_status[i] == StatusType::exceeded_maximum_sequence_size) {
      long_seq++;
      continue;
    } else if (entry_status[i] == StatusType::exceeded_maximum_sequences_per_poa) {  // NOLINT
      skipped_seq++;
      continue;
    } else if (entry_status[i] != StatusType::success) {
      std::cerr << "[CUDABatchProcessor::AddWindow] error: "
                << "could not add sequence to POA in batch "
                << cudapoa_batch_->batch_id() << std::endl;
      exit(1);
    }
    seq_added++;
  }
  seqs_added_per_window_.push_back(seq_added);

#ifndef NDEBUG
  if (long_seq > 0) {
    std::cerr << "Too long (" << long_seq << " / " << num_seqs << ")"
              << std::endl;
  }
  if (skipped_seq > 0) {
    std::cerr << "Skipped (" << skipped_seq << " / " << num_seqs << ")"
              << std::endl;
  }
#endif

  return true;
}

bool CUDABatchProcessor::HasWindows() const {
  return (cudapoa_batch_->get_total_poas() > 0);
}

void CUDABatchProcessor::GeneratePOA() {
  // call generate poa function
  cudapoa_batch_->generate_poa();
}

void CUDABatchProcessor::GetConsensus() {
  std::vector<std::string> consensuses;
  std::vector<std::vector<std::uint16_t>> coverages;
  std::vector<StatusType> output_status;
  cudapoa_batch_->get_consensus(consensuses, coverages, output_status);

  for (std::uint32_t i = 0; i < windows_.size(); i++) {
    auto window = windows_.at(i);
    if (output_status.at(i) == StatusType::success) {
      // This is a special case borrowed from the CPU version.
      // TODO(Nvidia): We still run this case through the GPU but could ommit it
      if (window->sequences_ids.size() < 2) {
        window->consensus = window->backbone;
      } else {
        window->consensus = consensuses[i];
        window->status = true;
        if (window->type ==  WindowType::kTGS) {
          std::uint32_t num_seqs_in_window = seqs_added_per_window_[i];
          std::uint32_t average_coverage = num_seqs_in_window / 2;

          std::int32_t begin = 0, end = window->consensus.size() - 1;
          for (; begin < static_cast<std::int32_t>(window->consensus.size()); ++begin) {  // NOLINT
            if (coverages[i][begin] >= average_coverage) {
              break;
            }
          }
          for (; end >= 0; --end) {
            if (coverages[i][end] >= average_coverage) {
              break;
            }
          }
          if (begin < end) {
            window->consensus = window->consensus.substr(begin, end - begin + 1);  // NOLINT
          }
        }
      }

      std::vector<uint32_t>{}.swap(window->sequences_ids);
      std::vector<std::pair<std::uint32_t, std::uint32_t>>{}.swap(window->sequences_intervals);  // NOLINT
      std::vector<std::pair<std::uint32_t, std::uint32_t>>{}.swap(window->positions);  // NOLINT
    }
  }
}

void CUDABatchProcessor::GenerateConsensus() {
  // Generate consensus for all windows in the batch
  GeneratePOA();
  GetConsensus();
  return;
}

void CUDABatchProcessor::Reset() {
  windows_.clear();
  seqs_added_per_window_.clear();
  cudapoa_batch_->reset();
}

}  // namespace racon
