/*!
 * @file cudapolisher.cpp
 *
 * @brief CUDA Polisher class source file
 */

#include "cudapolisher.hpp"

#include <cuda_profiler_api.h>

#include <chrono>  // NOLINT
#include <future>  // NOLINT
#include <iostream>
#include <utility>
#include <string>

#include "biosoup/progress_bar.hpp"
#include "biosoup/timer.hpp"
#include "claraparabricks/genomeworks/utils/cudautils.hpp"

namespace racon {

CUDAPolisher::CUDAPolisher(
    std::shared_ptr<thread_pool::ThreadPool> thread_pool,
    double q,
    double e,
    std::uint32_t w,
    bool trim,
    std::int8_t m,
    std::int8_t n,
    std::int8_t g,
    std::uint32_t cudapoa_batches, bool cuda_banded_alignment,
    std::uint32_t cudaaligner_batches,
    std::uint32_t cudaaligner_band_width)
    : Polisher(thread_pool, q, e, w, trim, m, n, g),
      cudapoa_batches_(cudapoa_batches),
      cudaaligner_batches_(cudaaligner_batches),
      gap_(g),
      mismatch_(n),
      match_(m),
      cuda_banded_alignment_(cuda_banded_alignment),
      cudaaligner_band_width_(cudaaligner_band_width) {
  claraparabricks::genomeworks::cudapoa::Init();
  claraparabricks::genomeworks::cudaaligner::Init();

  GW_CU_CHECK_ERR(cudaGetDeviceCount(&num_devices_));

  if (num_devices_ < 1) {
    throw std::runtime_error("No GPU devices found.");
  }

  // Run dummy call on each device to initialize CUDA context.
  for (std::int32_t dev_id = 0; dev_id < num_devices_; dev_id++) {
    GW_CU_CHECK_ERR(cudaSetDevice(dev_id));
    GW_CU_CHECK_ERR(cudaFree(0));
    std::cerr << "[racon::CUDAPolisher::CUDAPolisher] initialized device "
              << dev_id << std::endl;
  }
}

CUDAPolisher::~CUDAPolisher() {
  cudaDeviceSynchronize();
  cudaProfilerStop();
}

void CUDAPolisher::AllocateMemory(std::size_t step) {
  biosoup::Timer timer{};
  timer.Start();

  if (step == 0) {
    /* dummy */
  } else if (step == 1 && cudaaligner_batches_ > 0) {
    if (cudaaligner_band_width_ == 0) {
      cudaaligner_band_width_ = mean_overlap_len_ * 0.1f;
    }
    for (std::int32_t device = 0; device < num_devices_; device++) {
      GW_CU_CHECK_ERR(cudaSetDevice(device));

      std::size_t free, total;
      GW_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
      const std::size_t free_usable_memory =
          static_cast<float>(free) * 90 / 100;  // Using 90% of available memory
      const std::int64_t usable_memory_per_aligner =
          free_usable_memory / cudaaligner_batches_;
      const std::int32_t max_bandwidth =
          cudaaligner_band_width_ & ~0x1;  // Band width needs to be even

      for (std::uint32_t batch = 0; batch < cudaaligner_batches_; batch++) {
        batch_aligners_.emplace_back(CreateCUDABatchAligner(
            max_bandwidth,
            device,
            usable_memory_per_aligner));
      }
    }

    std::cerr << "[racon::CUDAPolisher::Polish] allocated GPU memory for alignment "  // NOLINT
              << timer.Stop() << "s"
              << std::endl;
  } else if (step == 2 && cudapoa_batches_ > 0) {
    // Creation and use of batches
    const std::uint32_t MAX_DEPTH_PER_WINDOW = 200;

    for (std::int32_t device = 0; device < num_devices_; device++) {
      std::size_t total = 0, free = 0;
      GW_CU_CHECK_ERR(cudaSetDevice(device));
      GW_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
      // Using 90% of available memory as heuristic since not all available
      // memory can be used due to fragmentation
      std::size_t mem_per_batch = 0.9 * free / cudapoa_batches_;
      for (std::uint32_t batch = 0; batch < cudapoa_batches_; batch++) {
        batch_processors_.emplace_back(CreateCUDABatch(
            MAX_DEPTH_PER_WINDOW,
            device,
            mem_per_batch,
            gap_,
            mismatch_,
            match_,
            cuda_banded_alignment_));
      }
    }

    std::cerr << "[racon::CUDAPolisher::Polish] allocated GPU memory for polishing "  // NOLINT
              << timer.Stop() << "s"
              << std::endl;
  }
}

void CUDAPolisher::DeallocateMemory(std::size_t step) {
  if (step == 0) {
    /* dummy */
  } else if (step == 1) {
    batch_aligners_.clear();
  } else if (step == 2) {
    batch_processors_.clear();
  }
}

void CUDAPolisher::FindIntervals(
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences,
    std::vector<Overlap>* overlaps) {
  if (cudaaligner_batches_ > 0) {
    biosoup::Timer timer{};
    timer.Start();

    std::mutex mutex_overlaps;
    std::uint32_t next_overlap_index = 0;

    // Lambda expression for filling up next batch of alignments
    auto fill_next_batch = [&] (CUDABatchAligner* batch) ->
        std::pair<uint32_t, uint32_t> {
      batch->Reset();

      // Use mutex to read the vector containing windows in a threadsafe manner
      std::lock_guard<std::mutex> guard(mutex_overlaps);

      std::uint32_t initial_count = next_overlap_index;
      std::uint32_t count = overlaps->size();
      while (next_overlap_index < count) {
        if (batch->AddOverlap(&(*overlaps)[next_overlap_index], targets, sequences)) {  // NOLINT
          next_overlap_index++;
        } else {
          break;
        }
      }
      return {initial_count, next_overlap_index};
    };

    std::mutex mutex_bar;
    biosoup::ProgressBar bar{static_cast<std::uint32_t>(overlaps->size()), 16};

    // Lambda expression for processing a batch of alignments
    auto process_batch = [&] (CUDABatchAligner* batch) -> void {
      while (true) {
        auto range = fill_next_batch(batch);
        auto updates = range.second - range.first;
        if (batch->HasOverlaps()) {
          // Launch workload
          batch->AlignAll();

          // Generate CIGAR strings for successful alignments. The actual
          // breaking points will be calculate by the overlap object
          batch->GenerateCigarStrings();
          for (; range.first < range.second; ++range.first) {
            if (!(*overlaps)[range.first].alignment.empty()) {
              (*overlaps)[range.first].FindIntervals(w_);
            }
          }

          {
            std::lock_guard<std::mutex> guard(mutex_bar);
            while (updates--) {
              if (++bar) {
                std::cerr << "[racon::CUDAPolisher::Polish] aligned "
                          << bar.event_counter() << " / "
                          << overlaps->size() << " overlaps "
                          << "[" << bar << "] "
                          << timer.Lap() << "s"
                          << "\r";
              }
            }
          }
        } else {
          break;
        }
      }
    };

    // Run batched alignment
    std::vector<std::future<void>> thread_futures;
    for (auto& aligner : batch_aligners_) {
      thread_futures.emplace_back(thread_pool_->Submit(
          process_batch,
          aligner.get()));
    }

    for (const auto& it : thread_futures) {
      it.wait();
    }
    std::cerr << std::endl;
  }

  // This call runs the breaking point detection code for all alignments.
  // Any overlaps that couldn't be processed by the GPU are also handled here
  // by the CPU aligner.
  Polisher::FindIntervals(targets, sequences, overlaps);
}

void CUDAPolisher::GenerateConsensus(
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences) {
  if (cudapoa_batches_ > 0) {
    biosoup::Timer timer{};
    timer.Start();

    // Mutex for accessing the vector of windows
    std::mutex mutex_windows;

    // Index of next window to be added to a batch
    std::uint32_t next_window_index = 0;

    // Lambda function for adding windows to batches
    auto fill_next_batch = [&] (CUDABatchProcessor* batch) -> std::uint32_t {
      batch->Reset();

      // Use mutex to read the vector containing windows in a threadsafe manner
      std::lock_guard<std::mutex> guard(mutex_windows);

      // TODO(Nvidia): Reducing window size by 10 for debugging.
      std::uint32_t count = 0;
      while (next_window_index < windows_.size()) {
        if (windows_[next_window_index]->consensus.empty() &&
            windows_[next_window_index]->sequences_ids.empty() == false) {
          if (!batch->AddWindow(windows_.at(next_window_index), sequences)) {
            break;
          }
          ++count;
        }
        ++next_window_index;
      }
      return count;
    };

    std::uint32_t num_windows = 0;
    for (const auto& it : windows_) {
      if (it->consensus.empty() && !it->sequences_ids.empty()) {
        ++num_windows;
      }
    }

    std::mutex mutex_bar;
    biosoup::ProgressBar bar{static_cast<std::uint32_t>(num_windows), 16};

    // Lambda function for processing each batch
    auto process_batch = [&] (CUDABatchProcessor* batch) -> void {
      while (true) {
        std::uint32_t updates = fill_next_batch(batch);
        if (batch->HasWindows()) {
          // Launch workload
          batch->GenerateConsensus();

          // progress bar
          {
            std::lock_guard<std::mutex> guard(mutex_bar);
            while (updates--) {
              if (++bar) {
                std::cerr << "[racon::CUDAPolisher::Polish] called consensus for "  // NOLINT
                          << bar.event_counter() << " / " << num_windows << " windows "  // NOLINT
                          << "[" << bar << "] "
                          << timer.Lap() << "s"
                          << "\r";
              }
            }
          }
        } else {
          break;
        }
      }
    };

    // Process each of the batches in a separate thread
    std::vector<std::future<void>> thread_futures;
    for (auto& batch_processor : batch_processors_) {
      thread_futures.emplace_back(thread_pool_->Submit(
          process_batch,
          batch_processor.get()));
    }

    for (const auto& it : thread_futures) {
      it.wait();
    }
    std::cerr << std::endl;
  }

  // This call runs the generates consensuses for all windows. Any windows that
  // couldn't be processed by the GPU are also handled here by the CPU
  Polisher::GenerateConsensus(sequences);
}

}  // namespace racon
