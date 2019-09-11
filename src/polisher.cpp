/*!
 * @file polisher.cpp
 *
 * @brief Polisher class source file
 */

#include <iostream>
#include <exception>
#include <algorithm>
#include <unordered_set>

#include "bioparser/bioparser.hpp"
#include "thread_pool/thread_pool.hpp"
#include "logger/logger.hpp"
#include "spoa/spoa.hpp"
#include "ram/ram.hpp"

#include "overlap.hpp"
#include "window.hpp"
#include "racon/polisher.hpp"
#ifdef CUDA_ENABLED
#include "cuda/cudapolisher.hpp"
#endif

namespace racon {

constexpr uint32_t kChunkSize = 1024 * 1024 * 1024; // ~ 1GB

template<class T>
uint64_t shrinkToFit(std::vector<std::unique_ptr<T>>& src, uint64_t begin) {

    uint64_t i = begin;
    for (uint64_t j = begin; i < src.size(); ++i) {
        if (src[i] != nullptr) {
            continue;
        }

        j = std::max(j, i);
        while (j < src.size() && src[j] == nullptr) {
            ++j;
        }

        if (j >= src.size()) {
            break;
        } else if (i != j) {
            src[i].swap(src[j]);
        }
    }
    uint64_t num_deletions = src.size() - i;
    if (i < src.size()) {
        src.resize(i);
    }
    return num_deletions;
}

std::unique_ptr<Polisher> createPolisher(std::uint32_t q, std::uint32_t e,
    std::uint32_t w, bool trim, std::int8_t m, std::int8_t n, std::int8_t g,
    std::shared_ptr<thread_pool::ThreadPool> thread_pool,
    std::uint32_t cudapoa_batches, bool cuda_banded_alignment,
    std::uint32_t cudaaligner_batches) {

    if (w == 0) {
        throw std::invalid_argument("[racon::createPolisher] error: "
            "invalid window length!");
    }
    if (g > 0) {
        throw std::invalid_argument("[racon::createPolisher] error: "
            "gap penalty must be non-positive!");
    }
    if (thread_pool == nullptr) {
        throw std::invalid_argument("[racon::createPolisher] error: "
            "thread_pool is nullptr!");
    }

    if (cudapoa_batches > 0 || cudaaligner_batches > 0)
    {
#ifdef CUDA_ENABLED
        // If CUDA is enabled, return an instance of the CUDAPolisher object.
        return std::unique_ptr<Polisher>(new CUDAPolisher(q, e, w, trim, m, n, g,
            thread_pool, cudapoa_batches, cuda_banded_alignment, cudaaligner_batches));
#else
        throw std::logic_error("[racon::createPolisher] error: "
            "Attemping to use CUDA when CUDA support is not available");
#endif
    }
    else
    {
        (void) cuda_banded_alignment;
        return std::unique_ptr<Polisher>(new Polisher(q, e, w, trim, m, n, g,
            thread_pool));
    }
}

Polisher::Polisher(std::uint32_t q, std::uint32_t e, std::uint32_t w, bool trim,
    std::int8_t m, std::int8_t n, std::int8_t g,
    std::shared_ptr<thread_pool::ThreadPool> thread_pool)
        : q_(q), e_(e), w_(w), trim_(trim), windows_(), alignment_engines_(),
        dummy_quality_(w, '!'), headers_(), thread_to_id_(),
        thread_pool_(thread_pool) {

    std::uint32_t id = 0;
    for (const auto& it: thread_pool_->thread_identifiers()) {
        thread_to_id_[it] = id++;
    }

    for (std::uint32_t i = 0; i < thread_pool_->num_threads(); ++i) {
        alignment_engines_.emplace_back(spoa::createAlignmentEngine(
            spoa::AlignmentType::kNW, m, n, g));
        alignment_engines_.back()->prealloc(w_, 5);
    }
}

Polisher::~Polisher() {
}

void Polisher::initialize(
    const std::vector<std::unique_ptr<ram::Sequence>>& targets,
    const std::vector<std::unique_ptr<ram::Sequence>>& sequences) {

    headers_.clear();
    windows_.clear();

    if (targets.empty() || sequences.empty()) {
        return;
    }

    for (const auto& it: targets) {
        headers_.emplace_back(new ram::Sequence());
        headers_.back()->id = 0;
        headers_.back()->name = it->name;
    }

    // ram::Overlap helper functions
    auto overlap_length = [] (const ram::Overlap& o) -> std::uint32_t {
        return std::max(o.t_end - o.t_begin, o.q_end - o.q_begin);
    };
    auto overlap_error = [&] (const ram::Overlap& o) -> double {
        return 1 - std::min(o.t_end - o.t_begin, o.q_end - o.q_begin) /
            static_cast<double>(overlap_length(o));
    };

    auto minimizer_engine = ram::createMinimizerEngine(15, 5, thread_pool_);
    std::vector<std::unique_ptr<Overlap>> overlaps(sequences.size());

    logger::Logger logger;

    for (std::uint32_t i = 0, j = 0, bytes = 0; i < targets.size(); ++i) {
        bytes += targets[i]->data.size();
        if (i != targets.size() - 1 && bytes < (1U << 30)) {
            continue;
        }
        bytes = 0;

        logger.log();

        minimizer_engine->minimize(targets.begin() + j, targets.begin() + i + 1);
        minimizer_engine->filter(0.001);

        logger.log("[racon::Polisher::initialize] minimized " + std::to_string(j) +
            " - " + std::to_string(i + 1) + " / " + std::to_string(targets.size()));
        logger.log();

        std::vector<std::future<std::vector<ram::Overlap>>> thread_futures;

        for (std::uint32_t k = 0; k < sequences.size(); ++k) {
            thread_futures.emplace_back(thread_pool_->submit(
                [&] (std::uint32_t i) -> std::vector<ram::Overlap> {
                    return minimizer_engine->map(sequences[i], false, false);
                }
            , k));

            bytes += sequences[k]->data.size();
            if (k != sequences.size() - 1 && bytes < (1U << 30)) {
                continue;
            }
            bytes = 0;

            for (auto& it: thread_futures) {
                for (const auto& jt: it.get()) {
                    if (overlap_error(jt) >= 0.3) {
                        continue;
                    }
                    if (overlaps[jt.q_id] == nullptr ||
                        overlaps[jt.q_id]->length() < overlap_length(jt)) {
                        overlaps[jt.q_id] = std::unique_ptr<racon::Overlap>(
                            new Overlap(jt.q_id, jt.q_begin, jt.q_end, jt.t_id,
                            jt.t_begin, jt.t_end, jt.strand));
                    }
                }
            }
            thread_futures.clear();
        }

        logger.log("[racon::Polisher::initialize] mapped sequences");

        j = i + 1;
    }

    for (std::uint32_t i = 0, j = 0; i < overlaps.size(); ++i) {
        if (overlaps[i] != nullptr) {
            if (overlaps[i]->strand == 0) {
                for (auto& it: sequences[overlaps[i]->q_id]->data) {
                    switch (it) {
                        case 'a': case 'A': it = 'T'; break;
                        case 'c': case 'C': it = 'G'; break;
                        case 'g': case 'G': it = 'C'; break;
                        case 't': case 'T': it = 'A'; break;
                        default: break;
                    }
                }
                std::reverse(
                    sequences[overlaps[i]->q_id]->data.begin(),
                    sequences[overlaps[i]->q_id]->data.end());
                std::reverse(
                    sequences[overlaps[i]->q_id]->quality.begin(),
                    sequences[overlaps[i]->q_id]->quality.end());
                std::uint32_t q_begin = overlaps[i]->q_begin;
                overlaps[i]->q_begin = sequences[overlaps[i]->q_id]->data.size() -
                    overlaps[i]->q_end;
                overlaps[i]->q_end = sequences[overlaps[i]->q_id]->data.size() -
                    q_begin;
            }
            overlaps[j++].swap(overlaps[i]);
        }
        if (i == overlaps.size() - 1) {
            overlaps.resize(j);
        }
    }

    find_break_points(overlaps, targets, sequences);

    logger.log();

    std::vector<std::uint64_t> id_to_window(targets.size() + 1, 0);
    for (std::uint64_t i = 0; i < targets.size(); ++i) {
        std::uint32_t k = 0;
        for (std::uint32_t j = 0; j < targets[i]->data.size(); j += w_, ++k) {

            std::uint32_t length = std::min(j + w_,
                static_cast<std::uint32_t>(targets[i]->data.size())) - j;

            windows_.emplace_back(createWindow(i, k, WindowType::kTGS,
                &(targets[i]->data[j]), length, targets[i]->quality.empty() ?
                &(dummy_quality_[0]) : &(targets[i]->quality[j]), length));
        }

        id_to_window[i + 1] = id_to_window[i] + k;
    }

    for (std::uint64_t i = 0; i < overlaps.size(); ++i) {

        ++headers_[overlaps[i]->t_id]->id;

        const auto& sequence = sequences[overlaps[i]->q_id];
        const auto& break_points = overlaps[i]->break_points;

        for (std::uint32_t j = 0; j < break_points.size(); j += 2) {
            if (break_points[j + 1].second - break_points[j].second < 0.02 * w_) {
                continue;
            }

            if (!sequence->quality.empty()) {

                double avg_q = 0;
                for (std::uint32_t k = break_points[j].second; k < break_points[j + 1].second; ++k) {
                    avg_q += static_cast<std::uint32_t>(sequence->quality[k]) - 33;
                }
                avg_q /= break_points[j + 1].second - break_points[j].second;

                if (avg_q < q_) {
                    continue;
                }
            }

            std::uint64_t window_id = id_to_window[overlaps[i]->t_id] +
                break_points[j].first / w_;
            std::uint32_t window_start = (break_points[j].first / w_) * w_;

            const char* data = &(sequence->data[break_points[j].second]);
            std::uint32_t data_length = break_points[j + 1].second -
                break_points[j].second;

            const char* quality = sequence->quality.empty() ? nullptr :
                &(sequence->quality[break_points[j].second]);
            std::uint32_t quality_length = quality == nullptr ? 0 : data_length;

            windows_[window_id]->add_layer(data, data_length,
                quality, quality_length,
                break_points[j].first - window_start,
                break_points[j + 1].first - window_start - 1);
        }

        overlaps[i].reset();
    }

    logger.log("[racon::Polisher::initialize] transformed data into windows");
}

void Polisher::find_break_points(
    std::vector<std::unique_ptr<Overlap>>& overlaps,
    const std::vector<std::unique_ptr<ram::Sequence>>& targets,
    const std::vector<std::unique_ptr<ram::Sequence>>& sequences) {

    logger::Logger logger;
    logger.log();

    std::vector<std::future<void>> thread_futures;
    for (std::uint64_t i = 0; i < overlaps.size(); ++i) {
        thread_futures.emplace_back(thread_pool_->submit(
            [&] (std::uint64_t i) -> void {
                overlaps[i]->find_break_points(targets, sequences, w_);
            }
        , i));
    }

    std::uint32_t logger_step = thread_futures.size() / 20;
    for (std::uint64_t i = 0; i < thread_futures.size(); ++i) {
        thread_futures[i].wait();
        if (logger_step != 0 && (i + 1) % logger_step == 0 && (i + 1) / logger_step < 20) {
            logger.bar("[racon::Polisher::initialize] aligning overlaps");
        }
    }
    if (logger_step != 0) {
        logger.bar("[racon::Polisher::initialize] aligning overlaps");
    } else {
        logger.log("[racon::Polisher::initialize] aligned overlaps");
    }
}

void Polisher::polish(std::vector<std::unique_ptr<ram::Sequence>>& dst,
    bool drop_unpolished) {

    logger::Logger logger;
    logger.log();

    std::vector<std::future<bool>> thread_futures;
    for (std::uint64_t i = 0; i < windows_.size(); ++i) {
        thread_futures.emplace_back(thread_pool_->submit(
            [&] (std::uint64_t i) -> bool {
                auto it = thread_to_id_.find(std::this_thread::get_id());
                return windows_[i]->generate_consensus(
                    alignment_engines_[it->second], trim_);
            }, i));
    }

    std::string polished_data = "";
    std::uint32_t num_polished_windows = 0;

    std::uint64_t logger_step = thread_futures.size() / 20;

    ram::Sequence::num_objects = 0;

    for (std::uint64_t i = 0; i < thread_futures.size(); ++i) {
        thread_futures[i].wait();

        num_polished_windows += thread_futures[i].get() == true ? 1 : 0;
        polished_data += windows_[i]->consensus();

        if (i == windows_.size() - 1 || windows_[i + 1]->rank() == 0) {
            double polished_ratio = num_polished_windows /
                static_cast<double>(windows_[i]->rank() + 1);

            if (!drop_unpolished || polished_ratio > 0) {
                std::string tags = " LN:i:" + std::to_string(polished_data.size());
                tags += " RC:i:" + std::to_string(headers_[windows_[i]->id()]->id);
                tags += " XC:f:" + std::to_string(polished_ratio);
                dst.emplace_back(new ram::Sequence(
                    headers_[windows_[i]->id()]->name + tags, polished_data));
            }

            num_polished_windows = 0;
            polished_data.clear();
        }
        windows_[i].reset();

        if (logger_step != 0 && (i + 1) % logger_step == 0 && (i + 1) / logger_step < 20) {
            logger.bar("[racon::Polisher::polish] generating consensus");
        }
    }

    if (logger_step != 0) {
        logger.bar("[racon::Polisher::polish] generating consensus");
    } else {
        logger.log("[racon::Polisher::polish] generated consensus");
    }
}

}
