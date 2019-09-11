/*!
 * @file polisher.hpp
 *
 * @brief Polisher class header file
 */

#pragma once

#include <stdlib.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <thread>

namespace thread_pool {
    class ThreadPool;
}

namespace spoa {
    class AlignmentEngine;
}

namespace ram {
    struct Sequence;
}

namespace racon {

class Overlap;
class Window;

class Polisher;
std::unique_ptr<Polisher> createPolisher(std::uint32_t q, std::uint32_t e,
    std::uint32_t w, bool trim, std::int8_t m, std::int8_t n, std::int8_t g,
    std::shared_ptr<thread_pool::ThreadPool> thread_pool,
    std::uint32_t cudapoa_batches = 0, bool cuda_banded_alignment = false,
    std::uint32_t cudaaligner_batches = 0);

class Polisher {
public:
    virtual ~Polisher();

    virtual void initialize(
        const std::vector<std::unique_ptr<ram::Sequence>>& targets,
        const std::vector<std::unique_ptr<ram::Sequence>>& sequences);

    virtual void polish(std::vector<std::unique_ptr<ram::Sequence>>& dst,
        bool drop_unpolished_sequences);

    friend std::unique_ptr<Polisher> createPolisher(std::uint32_t q, std::uint32_t e,
        std::uint32_t w, bool trim, std::int8_t m, std::int8_t n, std::int8_t g,
        std::shared_ptr<thread_pool::ThreadPool> thread_pool,
        std::uint32_t cudapoa_batches, bool cuda_banded_alignment,
        std::uint32_t cudaaligner_batches);
protected:
    Polisher(std::uint32_t q, std::uint32_t e, std::uint32_t w, bool trim,
        std::int8_t m, std::int8_t n, std::int8_t g,
        std::shared_ptr<thread_pool::ThreadPool> thread_pool);
    Polisher(const Polisher&) = delete;
    const Polisher& operator=(const Polisher&) = delete;

    virtual void find_break_points(
        std::vector<std::unique_ptr<Overlap>>& overlaps,
        const std::vector<std::unique_ptr<ram::Sequence>>& targets,
        const std::vector<std::unique_ptr<ram::Sequence>>& sequences);

    double q_;
    double e_;

    std::uint32_t w_;
    bool trim_;
    std::vector<std::shared_ptr<Window>> windows_;

    std::vector<std::shared_ptr<spoa::AlignmentEngine>> alignment_engines_;

    std::string dummy_quality_;
    std::vector<std::unique_ptr<ram::Sequence>> headers_;

    std::unordered_map<std::thread::id, uint32_t> thread_to_id_;
    std::shared_ptr<thread_pool::ThreadPool> thread_pool_;
};

}
