/*!
 * @file overlap.hpp
 *
 * @brief Overlap class header file
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <utility>

namespace ram {
    class Sequence;
}

namespace racon {

struct Overlap {

    Overlap(std::uint32_t q_id, std::uint32_t q_begin, std::uint32_t q_end,
        std::uint32_t t_id, std::uint32_t t_begin, std::uint32_t t_end,
        std::uint32_t strand);

    std::uint32_t length() const;

    void align(const char* q, std::uint32_t q_length, const char* t,
        std::uint32_t t_length);

    void find_break_points(
        const std::vector<std::unique_ptr<ram::Sequence>>& targets,
        const std::vector<std::unique_ptr<ram::Sequence>>& sequences,
        std::uint32_t w);
    void find_break_points(std::uint32_t w);

    std::uint32_t q_id;
    std::uint32_t q_begin;
    std::uint32_t q_end;
    std::uint32_t t_id;
    std::uint32_t t_begin;
    std::uint32_t t_end;
    std::uint32_t strand;
    std::string cigar;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> break_points;
};

}
