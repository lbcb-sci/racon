// Copyright (c) 2020 Robert Vaser

#ifndef RACON_POLISHER_HPP_
#define RACON_POLISHER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "biosoup/sequence.hpp"
#include "spoa/alignment_engine.hpp"
#include "thread_pool/thread_pool.hpp"

namespace racon {

struct Overlap;
struct Window;

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
      std::uint32_t cudaaligner_batches = 0,
      std::uint32_t cuda_aligner_band_width = 0);

  virtual std::vector<std::unique_ptr<biosoup::Sequence>> Polish(
      const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
      const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences,
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

  virtual void AllocateMemory(std::size_t /* step */) { /* dummy */ }

  virtual std::vector<Overlap> MapSequences(  // minimizers
      const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
      const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences);

  virtual void FindIntervals(  // global alignment
      const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
      const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences,
      std::vector<Overlap>* overlaps);

  virtual void GenerateConsensus();  // partial order alignment

  double q_;
  double e_;
  std::uint32_t w_;
  bool trim_;
  std::shared_ptr<thread_pool::ThreadPool> thread_pool_;
  std::uint32_t mean_overlap_len_;
  std::vector<std::unique_ptr<biosoup::Sequence>> headers_;
  std::string dummy_quality_;
  std::vector<std::shared_ptr<Window>> windows_;
  std::vector<std::shared_ptr<spoa::AlignmentEngine>> alignment_engines_;
};

}  // namespace racon

#endif  // RACON_POLISHER_HPP_
