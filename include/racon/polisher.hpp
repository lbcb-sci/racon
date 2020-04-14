/*!
 * @file polisher.hpp
 *
 * @brief Polisher class header file
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "biosoup/sequence.hpp"
#include "spoa/alignment_engine.hpp"
#include "thread_pool/thread_pool.hpp"

namespace racon {

class Overlap;
class Window;

class Polisher {
 public:
  Polisher(const Polisher&) = delete;
  Polisher& operator=(const Polisher&) = delete;

  Polisher(Polisher&&) = delete;
  Polisher& operator=(Polisher&&) = delete;

  virtual ~Polisher() = default;

  static std::unique_ptr<Polisher> Create(  // CPU or GPU polisher
      double quality_threshold,
      double error_threshold,
      std::uint32_t window_len,
      bool trim_consensus,
      std::int8_t match,
      std::int8_t mismatch,
      std::int8_t gap,
      std::shared_ptr<thread_pool::ThreadPool> thread_pool = nullptr,
      std::uint32_t cudapoa_batches = 0,
      bool cuda_banded_alignment = false,
      std::uint32_t cudaaligner_batches = 0);

  virtual void Initialize(
      const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
      const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences);

  virtual std::vector<std::unique_ptr<biosoup::Sequence>> Polish(
      bool drop_unpolished_sequences);

 protected:
  Polisher(
      double quality_threshold,
      double error_threshold,
      std::uint32_t window_len,
      bool trim_consensus,
      std::int8_t match,
      std::int8_t mismatch,
      std::int8_t gap,
      std::shared_ptr<thread_pool::ThreadPool> thread_pool);

  virtual void FindBreakPoints(
      const std::vector<std::unique_ptr<Overlap>>& overlaps,
      const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
      const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences);

  double q_;
  double e_;
  std::uint32_t w_;
  bool trim_;
  std::shared_ptr<thread_pool::ThreadPool> thread_pool_;
  std::vector<std::unique_ptr<biosoup::Sequence>> headers_;
  std::string dummy_quality_;
  std::vector<std::shared_ptr<Window>> windows_;
  std::vector<std::shared_ptr<spoa::AlignmentEngine>> alignment_engines_;
};

}  // namespace racon
