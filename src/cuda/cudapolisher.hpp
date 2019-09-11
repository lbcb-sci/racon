/*!
 * @file cudapolisher.hpp
 *
 * @brief CUDA Polisher class header file
 */

#pragma once

#include <mutex>

#include "racon/polisher.hpp"
#include "cudabatch.hpp"
#include "cudaaligner.hpp"
#include "thread_pool/thread_pool.hpp"


namespace racon {

class CUDAPolisher : public Polisher {
public:
    ~CUDAPolisher();

    virtual void polish(std::vector<std::unique_ptr<ram::Sequence>>& dst,
        bool drop_unpolished_sequences) override;

    friend std::unique_ptr<Polisher> createPolisher(std::uint32_t q, std::uint32_t e,
        std::uint32_t w, bool trim, std::int8_t m, std::int8_t n, std::int8_t g,
        std::shared_ptr<thread_pool::ThreadPool> thread_pool,
        std::uint32_t cudapoa_batches, bool cuda_banded_alignment,
        std::uint32_t cudaaligner_batches);
protected:
    CUDAPolisher(std::uint32_t q, std::uint32_t e, std::uint32_t w, bool trim,
        std::int8_t m, std::int8_t n, std::int8_t g,
        std::shared_ptr<thread_pool::ThreadPool> thread_pool,
        uint32_t cudapoa_batches, bool cuda_banded_alignment,
        uint32_t cudaaligner_batches);
    CUDAPolisher(const CUDAPolisher&) = delete;
    const CUDAPolisher& operator=(const CUDAPolisher&) = delete;

    virtual void find_break_points(
        std::vector<std::unique_ptr<Overlap>>& overlaps,
        const std::vector<std::unique_ptr<ram::Sequence>>& targets,
        const std::vector<std::unique_ptr<ram::Sequence>>& sequences) override;

    static std::vector<uint32_t> calculate_batches_per_gpu(uint32_t cudapoa_batches, uint32_t gpus);

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

}
