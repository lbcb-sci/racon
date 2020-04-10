/*!
 * @file cudapolisher.hpp
 *
 * @brief CUDA Polisher class header file
 */

#pragma once

#include <memory>
#include <mutex>  // NOLINT
#include <vector>

#include "cudabatch.hpp"
#include "cudaaligner.hpp"
#include "racon/polisher.hpp"

namespace racon {

class CUDAPolisher : public Polisher {
 public:
  ~CUDAPolisher();

  std::vector<std::unique_ptr<biosoup::Sequence>> Polish(
      bool drop_unpolished_sequences) override;

  friend Polisher;

 protected:
  CUDAPolisher(double q, double e, std::uint32_t w, bool trim,
    std::int8_t m, std::int8_t n, std::int8_t g,
    std::shared_ptr<thread_pool::ThreadPool> thread_pool,
    std::uint32_t cudapoa_batches, bool cuda_banded_alignment,
    std::uint32_t cudaaligner_batches);
  CUDAPolisher(const CUDAPolisher&) = delete;
  const CUDAPolisher& operator=(const CUDAPolisher&) = delete;

  void FindBreakPoints(
      const std::vector<std::unique_ptr<Overlap>>& overlaps,
      const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
      const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences) override;  // NOLINT

  static std::vector<uint32_t> calculate_batches_per_gpu(
      uint32_t cudapoa_batches, uint32_t gpus);

  // Vector of POA batches.
  std::vector<std::unique_ptr<CUDABatchProcessor>> batch_processors_;

  // Vector of aligner batches.
  std::vector<std::unique_ptr<CUDABatchAligner>> batch_aligners_;

  // Vector of bool indicating consensus generation status for each window.
  std::vector<bool> window_consensus_status_;

  // Number of batches for POA processing.
  uint32_t cudapoa_batches_;

  // Numbver of batches for Alignment processing.
  uint32_t cudaaligner_batches_;

  // Number of GPU devices to run with.
  int32_t num_devices_;

  // State transition scores.
  int8_t gap_;
  int8_t mismatch_;
  int8_t match_;

  // Use banded POA alignment
  bool cuda_banded_alignment_;
};

}  // namespace racon
