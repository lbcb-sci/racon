/*!
 * @file cudapolisher.hpp
 *
 * @brief CUDA Polisher class header file
 */

#ifndef RACON_CUDA_CUDAPOLISHER_HPP_
#define RACON_CUDA_CUDAPOLISHER_HPP_

#include <cstdint>
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

  friend Polisher;

 protected:
  CUDAPolisher(
      double q,
      double e,
      std::uint32_t w,
      bool trim,
      std::int8_t m,
      std::int8_t n,
      std::int8_t g,
      std::shared_ptr<thread_pool::ThreadPool> thread_pool,
      std::uint32_t cudapoa_batches,
      bool cuda_banded_alignment,
      std::uint32_t cudaaligner_batches,
      std::uint32_t cudaaligner_band_width);
  CUDAPolisher(const CUDAPolisher&) = delete;
  const CUDAPolisher& operator=(const CUDAPolisher&) = delete;

  void AllocateMemory(std::size_t step) override;

  void FindIntervals(
      const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
      const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences,
      std::vector<Overlap>* overlaps) override;

  void GenerateConsensus() override;

  static std::vector<std::uint32_t> CalculateBatchesPerGpu(
      std::uint32_t cudapoa_batches,
      std::uint32_t gpus);

  // Vector of POA batches.
  std::vector<std::unique_ptr<CUDABatchProcessor>> batch_processors_;

  // Vector of aligner batches.
  std::vector<std::unique_ptr<CUDABatchAligner>> batch_aligners_;

  // Number of batches for POA processing.
  std::uint32_t cudapoa_batches_;

  // Numbver of batches for Alignment processing.
  std::uint32_t cudaaligner_batches_;

  // Number of GPU devices to run with.
  std::int32_t num_devices_;

  // State transition scores.
  std::int8_t gap_;
  std::int8_t mismatch_;
  std::int8_t match_;

  // Use banded POA alignment
  bool cuda_banded_alignment_;

  // Band parameter for pairwise alignment
  std::uint32_t cudaaligner_band_width_;
};

}  // namespace racon

#endif  // RACON_CUDA_CUDAPOLISHER_HPP_
