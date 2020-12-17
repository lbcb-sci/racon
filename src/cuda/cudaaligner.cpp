/*!
 * @file cudaaligner.cpp
 *
 * @brief CUDABatchAligner class source file
 */

#include "cudaaligner.hpp"

#include <iostream>

#include "claraparabricks/genomeworks/utils/cudautils.hpp"

namespace racon {

using namespace claraparabricks::genomeworks::cudaaligner;  // NOLINT

std::atomic<uint32_t> CUDABatchAligner::batches;

std::unique_ptr<CUDABatchAligner> CreateCUDABatchAligner(
    std::uint32_t max_bandwidth,
    std::uint32_t device_id,
    std::int64_t max_gpu_memory) {
  return std::unique_ptr<CUDABatchAligner>(new CUDABatchAligner(
      max_bandwidth,
      device_id,
      max_gpu_memory));
}

CUDABatchAligner::CUDABatchAligner(
    std::uint32_t max_bandwidth,
    std::uint32_t device_id,
    std::int64_t max_gpu_memory)
    : overlaps_(),
      stream_(0) {
  bid_ = CUDABatchAligner::batches++;

  GW_CU_CHECK_ERR(cudaSetDevice(device_id));
  GW_CU_CHECK_ERR(cudaStreamCreate(&stream_));

  aligner_ = create_aligner(
      AlignmentType::global_alignment,
      max_bandwidth,
      stream_,
      device_id,
      max_gpu_memory);
}

CUDABatchAligner::~CUDABatchAligner() {
  aligner_.reset();
  GW_CU_CHECK_ERR(cudaStreamDestroy(stream_));
}

bool CUDABatchAligner::AddOverlap(
    Overlap* overlap,
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences) {
  std::string q = sequences[overlap->lhs_id]->InflateData(
      overlap->lhs_begin,
      overlap->lhs_end - overlap->lhs_begin);
  std::string t = targets[overlap->rhs_id]->InflateData(
      overlap->rhs_begin,
      overlap->rhs_end - overlap->rhs_begin);

  // NOTE: The cudaaligner API for adding alignments is the opposite of edlib.
  // Hence, what is treated as target in edlib is query in cudaaligner and
  // vice versa.
  StatusType s = aligner_->add_alignment(t.c_str(), t.size(), q.c_str(), q.size());  // NOLINT
  if (s == StatusType::exceeded_max_alignments) {
    return false;
  } else if (s == StatusType::exceeded_max_alignment_difference ||
             s == StatusType::exceeded_max_length) {
    // Do nothing as this case will be handled by CPU aligner.
  } else if (s != StatusType::success) {
    std::cerr << "Unknown error in cuda aligner!" << std::endl;
  } else {
    overlaps_.push_back(overlap);
  }
  return true;
}

void CUDABatchAligner::AlignAll() {
  aligner_->align_all();
}

void CUDABatchAligner::GenerateCigarStrings() {
  aligner_->sync_alignments();

  const std::vector<std::shared_ptr<Alignment>>& alignments =
      aligner_->get_alignments();
  // Number of alignments should be the same as number of overlaps.
  if (overlaps_.size() != alignments.size()) {
    throw std::runtime_error(
        "Number of alignments doesn't match number of overlaps in cudaaligner");
  }
  for (std::size_t a = 0; a < alignments.size(); a++) {
    overlaps_[a]->alignment = alignments[a]->convert_to_cigar();
  }
}

void CUDABatchAligner::Reset() {
  overlaps_.clear();
  cpu_overlap_data_.clear();
  aligner_->reset();
}

}  // namespace racon
