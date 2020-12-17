/*!
* @file cudabatch.hpp
 *
 * @brief CUDA batch class header file
 */

#ifndef RACON_CUDA_CUDABATCH_HPP_
#define RACON_CUDA_CUDABATCH_HPP_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <atomic>
#include <memory>
#include <vector>

#include "claraparabricks/genomeworks/cudapoa/batch.hpp"

#include "window.hpp"

namespace spoa {

class AlignmentEngine;

}

namespace racon {

class CUDABatchProcessor;
std::unique_ptr<CUDABatchProcessor> CreateCUDABatch(
    std::uint32_t max_window_depth,
    std::uint32_t device,
    std::size_t avail_mem,
    std::int8_t gap,
    std::int8_t mismatch,
    std::int8_t match,
    bool cuda_banded_alignment);

class CUDABatchProcessor {
 public:
  ~CUDABatchProcessor();

  /**
   * @brief Add a new window to the batch.
   *
   * @param[in] window : The window to add to the batch.
   *
   * @return True of window could be added to the batch.
   */
  bool AddWindow(
      std::shared_ptr<Window> window,
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences);

  /**
   * @brief Checks if batch has any windows to process.
   */
  bool HasWindows() const;

  /**
   * @brief Runs the core computation to generate consensus for
   *        all windows in the batch.
   */
  void GenerateConsensus();

  /**
   * @brief Resets the state of the object, which includes
   *        resetting buffer states and counters.
   */
  void Reset();

  /**
   * @brief Get batch ID.
   */
  std::uint32_t GetBatchID() const {
    return bid_;
  }

  // Builder function to create a new CUDABatchProcessor object.
  friend std::unique_ptr<CUDABatchProcessor> CreateCUDABatch(
    std::uint32_t max_window_depth,
    std::uint32_t device,
    std::size_t avail_mem,
    std::int8_t gap,
    std::int8_t mismatch,
    std::int8_t match,
    bool cuda_banded_alignment);

 protected:
  /**
   * @brief Constructor for CUDABatch class.
   *
   * @param[in] max_window_depth : Maximum number of sequences per window
   * @param[in] cuda_banded_alignment : Use banded POA alignment
   */
  CUDABatchProcessor(
      std::uint32_t max_window_depth,
      std::uint32_t device,
      std::size_t avail_mem,
      std::int8_t gap,
      std::int8_t mismatch,
      std::int8_t match,
      bool cuda_banded_alignment);
  CUDABatchProcessor(const CUDABatchProcessor&) = delete;
  const CUDABatchProcessor& operator=(const CUDABatchProcessor&) = delete;

  /*
   * @brief Run the CUDA kernel for generating POA on the batch.
   *        This call is asynchronous.
   */
  void GeneratePOA();

  /*
   * @brief Wait for execution to complete and grab the output
   *        consensus from the device.
   */
  void GetConsensus();

 protected:
  // Static batch count used to generate batch IDs.
  static std::atomic<std::uint32_t> batches;

  // Batch ID.
  std::uint32_t bid_ = 0;

  // CUDA-POA library object that manages POA batch.
  std::unique_ptr<claraparabricks::genomeworks::cudapoa::Batch> cudapoa_batch_;

  // Stream for running POA batch.
  cudaStream_t stream_;
  // Windows belonging to the batch.
  std::vector<std::shared_ptr<Window>> windows_;

  // Number of sequences actually added per window.
  std::vector<std::uint32_t> seqs_added_per_window_;
};

}  // namespace racon

#endif  // RACON_CUDA_CUDABATCH_HPP_
