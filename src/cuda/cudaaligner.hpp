/*!
* @file cudaaligner.hpp
 *
 * @brief CUDA aligner class header file
 */

#ifndef RACON_CUDA_CUDAALIGNER_HPP_
#define RACON_CUDA_CUDAALIGNER_HPP_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "biosoup/nucleic_acid.hpp"
#include "claraparabricks/genomeworks/cudaaligner/cudaaligner.hpp"
#include "claraparabricks/genomeworks/cudaaligner/aligner.hpp"
#include "claraparabricks/genomeworks/cudaaligner/alignment.hpp"

#include "overlap.hpp"

namespace racon {

class CUDABatchAligner;
std::unique_ptr<CUDABatchAligner> CreateCUDABatchAligner(
    std::uint32_t max_bandwidth,
    std::uint32_t device_id,
    std::int64_t max_gpu_memory);

class CUDABatchAligner {
 public:
  virtual ~CUDABatchAligner();

  /**
   * @brief Add a new overlap to the batch.
   *
   * @param[in] window   : The overlap to add to the batch.
   * @param[in] targets: Reference to a database of sequences.
   * @param[in] sequences: Reference to a database of sequences.
   *
   * @return True if overlap could be added to the batch.
   */
  virtual bool AddOverlap(
      Overlap* overlap,
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences);

  /**
   * @brief Checks if batch has any overlaps to process.
   *
   * @return Trie if there are overlaps in the batch.
   */
  virtual bool HasOverlaps() const {
    return overlaps_.size() > 0;
  }

  /**
   * @brief Runs batched alignment of overlaps on GPU.
   *
   */
  virtual void AlignAll();

  /**
   * @brief Generate cigar strings for overlaps that were successfully
   *        copmuted on the GPU.
   *
   */
  virtual void GenerateCigarStrings();

  /**
   * @brief Resets the state of the object, which includes
   *        resetting buffer states and counters.
   */
  virtual void Reset();

  /**
   * @brief Get batch ID.
   */
  uint32_t GetBatchID() const {
    return bid_;
  }

  // Builder function to create a new CUDABatchAligner object.
  friend std::unique_ptr<CUDABatchAligner>CreateCUDABatchAligner(
      std::uint32_t max_bandwidth,
      std::uint32_t device_id,
      std::int64_t max_gpu_memory);

 protected:
  CUDABatchAligner(
      std::uint32_t max_bandwidth,
      std::uint32_t device_id,
      std::int64_t max_gpu_memory);
  CUDABatchAligner(const CUDABatchAligner&) = delete;
  const CUDABatchAligner& operator=(const CUDABatchAligner&) = delete;

  std::unique_ptr<claraparabricks::genomeworks::cudaaligner::Aligner> aligner_;

  std::vector<Overlap*> overlaps_;

  std::vector<std::pair<std::string, std::string>> cpu_overlap_data_;

  // Static batch count used to generate batch IDs.
  static std::atomic<std::uint32_t> batches;

  // Batch ID.
  std::uint32_t bid_ = 0;

  // CUDA stream for batch.
  cudaStream_t stream_;
};

}  // namespace racon

#endif  // RACON_CUDA_CUDAALIGNER_HPP_
