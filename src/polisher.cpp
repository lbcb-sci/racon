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
    double quality_threshold,
    double error_threshold,
    std::uint32_t window_len,
    bool trim_consensus,
    std::int8_t match,
    std::int8_t mismatch,
    std::int8_t gap,
    std::shared_ptr<thread_pool::ThreadPool> thread_pool)
    : q_(quality_threshold),
      e_(error_threshold),
      w_(window_len),
      trim_(trim_consensus),
      thread_pool_(thread_pool ?
          thread_pool :
          std::make_shared<thread_pool::ThreadPool>(1)),
      mean_overlap_len_(0),
      headers_(),
      dummy_quality_(w_, '!'),
      windows_(),
      alignment_engines_() {
  for (std::uint32_t i = 0; i < thread_pool->num_threads(); ++i) {
    alignment_engines_.emplace_back(spoa::AlignmentEngine::Create(
        spoa::AlignmentType::kNW,
        match,
        mismatch,
        gap));
    alignment_engines_.back()->Prealloc(w_, 5);
  }
}

std::unique_ptr<Polisher> Polisher::Create(
    double quality_threshold,
    double error_threshold,
    std::uint32_t window_len,
    bool trim_consensus,
    std::int8_t match,
    std::int8_t mismatch,
    std::int8_t gap,
    std::shared_ptr<thread_pool::ThreadPool> thread_pool,
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
        quality_threshold,
        error_threshold,
        window_len,
        trim_consensus,
        match,
        mismatch,
        gap,
        thread_pool,
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
        quality_threshold,
        error_threshold,
        window_len,
        trim_consensus,
        match,
        mismatch,
        gap,
        thread_pool));
  }
}

std::vector<std::unique_ptr<biosoup::Sequence>> Polisher::Polish(
    const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
    const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences,
    bool drop_unpolished) {

  headers_.clear();
  windows_.clear();

  if (targets.empty() || sequences.empty()) {
    return std::vector<std::unique_ptr<biosoup::Sequence>>{};
  }

  for (const auto& it : targets) {
    headers_.emplace_back(new biosoup::Sequence());
    headers_.back()->name = it->name;
  }

  biosoup::Timer timer{};
  timer.Start();

  AllocateMemory(0);
  auto overlaps = MapSequences(targets, sequences);

  timer.Stop();
  timer.Start();

  for (std::uint32_t i = 0; i < overlaps.size(); ++i) {
    if (overlaps[i].strand == 0) {
      sequences[overlaps[i].lhs_id]->ReverseAndComplement();
      auto lhs_begin = overlaps[i].lhs_begin;
      auto lhs_len = sequences[overlaps[i].lhs_id]->data.size();
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

  timer.Stop();
  timer.Start();

  AllocateMemory(2);

  timer.Stop();
  timer.Start();

  std::vector<std::uint64_t> id_to_window(targets.size() + 1, 0);
  for (std::uint64_t i = 0; i < targets.size(); ++i) {
    std::uint32_t k = 0;
    for (std::uint32_t j = 0; j < targets[i]->data.size(); j += w_, ++k) {
      std::uint32_t length = std::min(
          static_cast<std::uint32_t>(targets[i]->data.size()) - j,
          w_);

      windows_.emplace_back(std::make_shared<Window>(
          i,
          k,
          WindowType::kTGS,
          &(targets[i]->data[j]),
          length,
          targets[i]->quality.empty() ?
              &(dummy_quality_[0]) :
              &(targets[i]->quality[j]),
          length));
    }
    id_to_window[i + 1] = id_to_window[i] + k;
  }
  for (std::uint64_t i = 0; i < overlaps.size(); ++i) {
    ++headers_[overlaps[i].rhs_id]->id;

    const auto& sequence = sequences[overlaps[i].lhs_id];
    const auto& intervals = overlaps[i].intervals;

    for (std::uint32_t j = 0; j < intervals.size(); j += 2) {
      if (intervals[j + 1].second - intervals[j].second < 0.02 * w_) {
        continue;
      }

      if (!sequence->quality.empty()) {
        double avg_q = 0;
        for (std::uint32_t k = intervals[j].second; k < intervals[j + 1].second; ++k) {  // NOLINT
          avg_q += static_cast<std::uint32_t>(sequence->quality[k]) - 33;
        }
        avg_q /= intervals[j + 1].second - intervals[j].second;

        if (avg_q < q_) {
          continue;
        }
      }

      std::uint64_t window_id =
          id_to_window[overlaps[i].rhs_id] + intervals[j].first / w_;
      std::uint32_t window_start = (intervals[j].first / w_) * w_;

      const char* data = &(sequence->data[intervals[j].second]);
      std::uint32_t data_length = intervals[j + 1].second - intervals[j].second;

      const char* quality = sequence->quality.empty() ?
          nullptr :
          &(sequence->quality[intervals[j].second]);
      std::uint32_t quality_length = quality == nullptr ? 0 : data_length;

      windows_[window_id]->AddLayer(
          data, data_length,
          quality, quality_length,
          intervals[j].first - window_start,
          intervals[j + 1].first - window_start - 1);
    }
  }

  std::cerr << "[racon::Polisher::Polish] transformed data into windows "
            << std::fixed << timer.Stop() << "s"
            << std::endl;

  timer.Start();

  GenerateConsensus();

  std::string polished_data = "";
  std::uint32_t num_polished_windows = 0;

  biosoup::Sequence::num_objects = 0;
  std::vector<std::unique_ptr<biosoup::Sequence>> dst;

  for (std::uint64_t i = 0; i < windows_.size(); ++i) {
    num_polished_windows += windows_[i]->status;
    polished_data += windows_[i]->consensus;

    if (i == windows_.size() - 1 || windows_[i + 1]->rank == 0) {
      double polished_ratio =
          num_polished_windows / static_cast<double>(windows_[i]->rank + 1);

      if (!drop_unpolished || polished_ratio > 0) {
        std::string tags = " LN:i:" + std::to_string(polished_data.size());
        tags += " RC:i:" + std::to_string(headers_[windows_[i]->id]->id);
        tags += " XC:f:" + std::to_string(polished_ratio);
        dst.emplace_back(new biosoup::Sequence(
            headers_[windows_[i]->id]->name + tags,
            polished_data));
      }

      num_polished_windows = 0;
      polished_data.clear();
    }
    windows_[i].reset();
  }

  timer.Stop();

  std::cerr << "[racon::Polisher::Polish] "
            << std::fixed <<  timer.elapsed_time() << "s"
            << std::endl;

  return dst;
}

std::vector<Overlap> Polisher::MapSequences(
    const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
    const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences) {
  // biosoup::Overlap helper functions
  auto overlap_length = [] (const biosoup::Overlap& o) -> std::uint32_t {
    return std::max(o.rhs_end - o.rhs_begin, o.lhs_end - o.lhs_begin);
  };
  auto overlap_error = [&] (const biosoup::Overlap& o) -> double {
    return 1 - std::min(o.rhs_end - o.rhs_begin, o.lhs_end - o.lhs_begin) /
        static_cast<double>(overlap_length(o));
  };
  // biosoup::Overlap helper functions

  auto minimizer_engine = ram::MinimizerEngine(15, 5, thread_pool_);
  std::vector<Overlap> overlaps(sequences.size());

  biosoup::Timer timer{};

  for (std::uint32_t i = 0, j = 0, bytes = 0; i < targets.size(); ++i) {
    bytes += targets[i]->data.size();
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

      bytes += sequences[k]->data.size();
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
    const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
    const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences,
    std::vector<Overlap>* overlaps) {
  biosoup::Timer timer{};
  timer.Start();

  std::vector<std::future<void>> futures;
  for (std::uint64_t i = 0; i < overlaps->size(); ++i) {
    if (!(*overlaps)[i].intervals.empty()) {
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

void Polisher::GenerateConsensus() {
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
        windows_[i]->GenerateConsensus(alignment_engines_[it->second], trim_);
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
