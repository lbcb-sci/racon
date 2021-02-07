// Copyright (c) Robert Vaser 2020

#include "racon/polisher.hpp"

#include <iostream>
#include <exception>
#include <algorithm>

#include "biosoup/progress_bar.hpp"
#include "biosoup/timer.hpp"
#include "ram/minimizer_engine.hpp"

#include "overlap.hpp"
#include "window.hpp"

#ifdef CUDA_ENABLED
#include "cuda/cudapolisher.hpp"
#endif

namespace racon {

Polisher::Polisher(
    std::shared_ptr<thread_pool::ThreadPool> thread_pool,
    double quality_threshold,
    double error_threshold,
    std::uint32_t window_len,
    bool trim_consensus,
    std::int8_t match,
    std::int8_t mismatch,
    std::int8_t gap)
    : q_(quality_threshold),
      e_(error_threshold),
      w_(window_len),
      trim_(trim_consensus),
      thread_pool_(thread_pool ?
          thread_pool :
          std::make_shared<thread_pool::ThreadPool>(1)),
      mean_overlap_len_(0),
      windows_(),
      alignment_engines_() {
  for (std::uint32_t i = 0; i < thread_pool_->num_threads(); ++i) {
    alignment_engines_.emplace_back(spoa::AlignmentEngine::Create(
        spoa::AlignmentType::kNW,
        match,
        mismatch,
        gap));
    alignment_engines_.back()->Prealloc(w_, 5);
  }
}

std::unique_ptr<Polisher> Polisher::Create(
    std::shared_ptr<thread_pool::ThreadPool> thread_pool,
    double quality_threshold,
    double error_threshold,
    std::uint32_t window_len,
    bool trim_consensus,
    std::int8_t match,
    std::int8_t mismatch,
    std::int8_t gap,
    std::uint32_t cudapoa_batches,
    bool cuda_banded_alignment,
    std::uint32_t cudaaligner_batches,
    std::uint32_t cuda_aligner_band_width) {

  if (window_len == 0) {
    throw std::invalid_argument(
        "[racon::Polisher::Create] error: invalid window length");
  }
  if (gap > 0) {
    throw std::invalid_argument(
        "[racon::Polisher::Create] error: gap penalty must be non-positive");
  }

  if (cudapoa_batches > 0 || cudaaligner_batches > 0) {
#ifdef CUDA_ENABLED
    // If CUDA is enabled, return an instance of the CUDAPolisher object.
    return std::unique_ptr<Polisher>(new CUDAPolisher(
        thread_pool,
        quality_threshold,
        error_threshold,
        window_len,
        trim_consensus,
        match,
        mismatch,
        gap,
        cudapoa_batches,
        cuda_banded_alignment,
        cudaaligner_batches,
        cuda_aligner_band_width));
#else
    throw std::logic_error(
        "[racon::Polisher::Create] error: CUDA support is not available");
#endif
  } else {
    (void) cuda_banded_alignment;
    (void) cuda_aligner_band_width;
    return std::unique_ptr<Polisher>(new Polisher(
        thread_pool,
        quality_threshold,
        error_threshold,
        window_len,
        trim_consensus,
        match,
        mismatch,
        gap));
  }
}

std::vector<std::unique_ptr<biosoup::NucleicAcid>> Polisher::Polish(
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences,
    bool drop_unpolished) {

  if (targets.empty() || sequences.empty()) {
    return std::vector<std::unique_ptr<biosoup::NucleicAcid>>{};
  }

  biosoup::Timer timer{};
  timer.Start();

  AllocateMemory(0);
  auto overlaps = MapSequences(targets, sequences);
  DeallocateMemory(0);

  std::cerr << "[racon::Polisher::Polish] found " << overlaps.size() << " overlaps "  // NOLINT
            << std::fixed << timer.Stop() << "s"
            << std::endl;

  timer.Start();

  for (std::uint32_t i = 0; i < overlaps.size(); ++i) {
    if (overlaps[i].strand == 0) {
      sequences[overlaps[i].lhs_id]->ReverseAndComplement();
      auto lhs_begin = overlaps[i].lhs_begin;
      auto lhs_len = sequences[overlaps[i].lhs_id]->inflated_len;
      overlaps[i].lhs_begin = lhs_len - overlaps[i].lhs_end;
      overlaps[i].lhs_end = lhs_len - lhs_begin;
    }
  }

  std::cerr << "[racon::Polisher::Polish] reverse complemented sequences "
            << std::fixed << timer.Stop() << "s"
            << std::endl;

  timer.Start();

  AllocateMemory(1);
  FindIntervals(targets, sequences, &overlaps);
  DeallocateMemory(1);

  timer.Stop();
  timer.Start();

  std::vector<std::uint64_t> target_to_window(targets.size() + 1, 0);
  for (std::uint64_t i = 0; i < targets.size(); ++i) {
    for (std::uint32_t j = 0; j < targets[i]->inflated_len; j += w_) {
      windows_.emplace_back(std::make_shared<Window>(
          i,
          windows_.size() - target_to_window[i],
          WindowType::kTGS,
          targets[i]->InflateData(j, w_)));
    }
    target_to_window[i + 1] = windows_.size();
  }

  std::vector<std::uint32_t> num_reads(targets.size(), 0);
  for (std::uint64_t i = 0; i < overlaps.size(); ++i) {
    ++num_reads[overlaps[i].rhs_id];
    for (std::uint32_t j = 0; j < overlaps[i].lhs_intervals.size(); ++j) {
      if (overlaps[i].lhs_intervals[j].second -
          overlaps[i].lhs_intervals[j].first < .02 * w_) {
        continue;
      }
      if (!sequences[overlaps[i].lhs_id]->block_quality.empty()) {
        double avg_q = 0;
        std::uint32_t k = overlaps[i].lhs_intervals[j].first;
        for (; k < overlaps[i].lhs_intervals[j].second; ++k) {
          avg_q += sequences[overlaps[i].lhs_id]->Score(k);
        }
        avg_q /= overlaps[i].lhs_intervals[j].second - overlaps[i].lhs_intervals[j].first;  // NOLINT
        if (avg_q < q_) {
          continue;
        }
      }

      std::uint32_t rank = overlaps[i].rhs_intervals[j].first / w_;

      windows_[target_to_window[overlaps[i].rhs_id] + rank]->AddLayer(
          overlaps[i].lhs_id,
          overlaps[i].lhs_intervals[j].first,
          overlaps[i].lhs_intervals[j].second,
          overlaps[i].rhs_intervals[j].first - rank * w_,
          overlaps[i].rhs_intervals[j].second - rank * w_);
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>>{}.swap(overlaps[i].lhs_intervals);  // NOLINT
    std::vector<std::pair<std::uint32_t, std::uint32_t>>{}.swap(overlaps[i].rhs_intervals);  // NOLINT
  }
  std::vector<racon::Overlap>{}.swap(overlaps);

  std::cerr << "[racon::Polisher::Polish] prepared " << windows_.size()
            << " window placeholders "
            << std::fixed << timer.Stop() << "s"
            << std::endl;

  timer.Start();

  AllocateMemory(2);
  GenerateConsensus(sequences);
  DeallocateMemory(2);

  timer.Stop();
  timer.Start();

  std::string polished_data = "";
  std::uint32_t num_polished_windows = 0;

  biosoup::NucleicAcid::num_objects = 0;
  std::vector<std::unique_ptr<biosoup::NucleicAcid>> dst;

  for (std::uint64_t i = 0, j = 0; i < windows_.size(); ++i) {
    num_polished_windows += windows_[i]->status;
    polished_data += windows_[i]->consensus;

    if (i == windows_.size() - 1 || windows_[i + 1]->rank == 0) {
      double polished_ratio =
          num_polished_windows / static_cast<double>(windows_[i]->rank + 1);

      if (!drop_unpolished || polished_ratio > 0) {
        std::string tags = " LN:i:" + std::to_string(polished_data.size());
        tags += " RC:i:" + std::to_string(num_reads[j]);
        tags += " XC:f:" + std::to_string(polished_ratio);
        dst.emplace_back(new biosoup::NucleicAcid(
            targets[j]->name + tags,
            polished_data));
        ++j;
      }

      num_polished_windows = 0;
      polished_data.clear();
    }
    windows_[i].reset();
  }
  windows_.clear();

  timer.Stop();

  std::cerr << "[racon::Polisher::Polish] "
            << std::fixed <<  timer.elapsed_time() << "s"
            << std::endl;

  return dst;
}

std::vector<Overlap> Polisher::MapSequences(
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences) {
  // biosoup::Overlap helper functions
  auto overlap_length = [] (const biosoup::Overlap& o) -> std::uint32_t {
    return std::max(o.rhs_end - o.rhs_begin, o.lhs_end - o.lhs_begin);
  };
  auto overlap_error = [&] (const biosoup::Overlap& o) -> double {
    return 1 - std::min(o.rhs_end - o.rhs_begin, o.lhs_end - o.lhs_begin) /
        static_cast<double>(overlap_length(o));
  };
  // biosoup::Overlap helper functions

  auto minimizer_engine = ram::MinimizerEngine(thread_pool_);
  std::vector<Overlap> overlaps(sequences.size());

  biosoup::Timer timer{};

  for (std::uint32_t i = 0, j = 0, bytes = 0; i < targets.size(); ++i) {
    bytes += targets[i]->inflated_len;
    if (i != targets.size() - 1 && bytes < (1U << 30)) {
      continue;
    }
    bytes = 0;

    timer.Start();

    minimizer_engine.Minimize(targets.begin() + j, targets.begin() + i + 1);
    minimizer_engine.Filter(0.001);

    std::cerr << "[racon::Polisher::Polish]"
              << " minimized " << std::to_string(j)
              << " - " << std::to_string(i + 1)
              <<  " / " << std::to_string(targets.size()) << " "
              << std::fixed << timer.Stop() << "s"
              << std::endl;

    timer.Start();

    std::vector<std::future<std::vector<biosoup::Overlap>>> futures;

    for (std::uint32_t k = 0; k < sequences.size(); ++k) {
      futures.emplace_back(thread_pool_->Submit(
          [&] (std::uint32_t i) -> std::vector<biosoup::Overlap> {
            return minimizer_engine.Map(sequences[i], false, false);
          },
          k));

      bytes += sequences[k]->inflated_len;
      if (k != sequences.size() - 1 && bytes < (1U << 30)) {
        continue;
      }
      bytes = 0;

      for (auto& it : futures) {
        for (const auto& jt : it.get()) {
          if (overlap_error(jt) >= e_) {
            continue;
          }
          if (overlaps[jt.lhs_id].length() < overlap_length(jt)) {
            overlaps[jt.lhs_id] = jt;
          }
        }
      }
      futures.clear();
    }

    std::cerr << "[racon::Polisher::Polish] mapped sequences "
              << std::fixed << timer.Stop() << "s"
              << std::endl;

    j = i + 1;
  }

  // remove invalid overlaps
  for (std::uint32_t i = 0, j = 0; i < overlaps.size(); ++i) {
    if (overlaps[i].length() > 0) {
      std::swap(overlaps[j++], overlaps[i]);
    }
    if (i == overlaps.size() - 1) {
      overlaps.resize(j);
    }
  }

  // CUDAPolisher
  for (const auto& it : overlaps) {
    mean_overlap_len_ += it.length();
  }
  mean_overlap_len_ /= overlaps.size();
  // CUDAPolisher

  return overlaps;
}

void Polisher::FindIntervals(
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences,
    std::vector<Overlap>* overlaps) {
  biosoup::Timer timer{};
  timer.Start();

  std::vector<std::future<void>> futures;
  for (std::uint64_t i = 0; i < overlaps->size(); ++i) {
    if (!(*overlaps)[i].lhs_intervals.empty()) {
      continue;
    }
    futures.emplace_back(thread_pool_->Submit(
        [&] (std::uint64_t i) -> void {
          (*overlaps)[i].Align(sequences, targets);
          (*overlaps)[i].FindIntervals(w_);
        },
        i));
  }
  if (futures.empty()) {
    return;
  }

  biosoup::ProgressBar bar{static_cast<std::uint32_t>(futures.size()), 16};
  for (const auto& it : futures) {
    it.wait();
    if (++bar) {
      std::cerr << "[racon::Polisher::Polish] "
                << "aligned " << bar.event_counter() << " / "
                << futures.size() << " overlaps "
                << "[" << bar << "] "
                << std::fixed << timer.Lap() << "s"
                << "\r";
    }
  }
  std::cerr << std::endl;
}

void Polisher::GenerateConsensus(
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences) {
  biosoup::Timer timer{};
  timer.Start();

  std::vector<std::future<void>> futures;
  for (std::uint64_t i = 0; i < windows_.size(); ++i) {
    if (!windows_[i]->consensus.empty()) {
      continue;
    }
    futures.emplace_back(thread_pool_->Submit(
      [&] (std::uint64_t i) -> void {
        auto it = thread_pool_->thread_ids().find(std::this_thread::get_id());
        windows_[i]->GenerateConsensus(
            sequences,
            alignment_engines_[it->second],
            trim_);
      },
      i));
  }

  if (futures.empty()) {
    return;
  }

  biosoup::ProgressBar bar{static_cast<std::uint32_t>(futures.size()), 16};
  for (const auto& it : futures) {
    it.wait();
    if (++bar) {
      std::cerr << "[racon::Polisher::Polish] called consensus for "
                << bar.event_counter() << " / " << futures.size() << " windows "
                << "[" << bar << "] "
                << std::fixed << timer.Lap() << "s"
                << "\r";
    }
  }
  std::cerr << std::endl;
}

}  // namespace racon
