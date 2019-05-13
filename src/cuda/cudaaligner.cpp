/*!
 * @file cudaaligner.cpp
 *
 * @brief CUDABatchAligner class source file
 */

#include <cudautils/cudautils.hpp>

#include "cudaaligner.hpp"

namespace racon {

std::atomic<uint32_t> CUDABatchAligner::batches;

std::unique_ptr<CUDABatchAligner> createCUDABatchAligner(uint32_t max_query_size,
                                                         uint32_t max_target_size,
                                                         uint32_t max_alignments,
                                                         uint32_t device_id)
{
    return std::unique_ptr<CUDABatchAligner>(new CUDABatchAligner(max_query_size,
                                                                  max_target_size,
                                                                  max_alignments,
                                                                  device_id));
}

CUDABatchAligner::CUDABatchAligner(uint32_t max_query_size,
                                   uint32_t max_target_size,
                                   uint32_t max_alignments,
                                   uint32_t device_id)
    : aligner_(genomeworks::cudaaligner::create_aligner(max_query_size,
                                                        max_target_size,
                                                        max_alignments,
                                                        genomeworks::cudaaligner::AlignmentType::global,
                                                        device_id))
    , overlaps_()
    , stream_(0)
{
    bid_ = CUDABatchAligner::batches++;

    GW_CU_CHECK_ERR(cudaStreamCreate(&stream_));

    aligner_->set_cuda_stream(stream_);
}

CUDABatchAligner::~CUDABatchAligner()
{
    GW_CU_CHECK_ERR(cudaStreamDestroy(stream_));
}

bool CUDABatchAligner::addOverlap(Overlap* overlap, std::vector<std::unique_ptr<Sequence>>& sequences)
{
    const char* q = !overlap->strand_ ? &(sequences[overlap->q_id_]->data()[overlap->q_begin_]) :
        &(sequences[overlap->q_id_]->reverse_complement()[overlap->q_length_ - overlap->q_end_]);
    const char* t = &(sequences[overlap->t_id_]->data()[overlap->t_begin_]);

    genomeworks::cudaaligner::StatusType s = 
        aligner_->add_alignment(q, overlap->q_end_ - overlap->q_begin_,
                                t, overlap->t_end_ - overlap->t_begin_);
    if (s == genomeworks::cudaaligner::StatusType::exceeded_max_alignments)
    {
        return false;
    }
    else if (s != genomeworks::cudaaligner::StatusType::success)
    {
        fprintf(stderr, "Exceeded maximum length of string.\n");
    }
    else
    {
        overlaps_.push_back(overlap);
    }
    return true;
}

void CUDABatchAligner::alignAll()
{
    aligner_->align_all();
}

void CUDABatchAligner::find_breaking_points(uint32_t window_length)
{
    const std::vector<std::shared_ptr<genomeworks::cudaaligner::Alignment>>& alignments = aligner_->get_alignments();
    // Number of alignments should be the same as number of overlaps.
    if (overlaps_.size() != alignments.size())
    {
        throw std::runtime_error("Number of alignments doesn't match number of overlaps in cudaaligner.");
    }
    for(std::size_t a = 0; a < alignments.size(); a++)
    {
        overlaps_[a]->cigar_ = alignments[a]->convert_to_cigar();
        overlaps_[a]->find_breaking_points_from_cigar(window_length);
    }
}

void CUDABatchAligner::reset()
{
    overlaps_.clear();
    aligner_->reset();
}

}
