// Copyright (c) 2020 Robert Vaser

#ifndef RACON_POLISHER_HPP_
#define RACON_POLISHER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "biosoup/nucleic_acid.hpp"
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
      std::shared_ptr<thread_pool::ThreadPool> thread_pool = nullptr,
      std::uint64_t batch_size = 1ULL << 36,  // bytes
      double error_threshold = .3,
      std::uint32_t window_len = 500,
      bool trim_consensus = true,
      std::int8_t match = 3,
      std::int8_t mismatch = -5,
      std::int8_t gap = -4,
      std::uint32_t cudapoa_batches = 0,
      bool cuda_banded_alignment = false,
      std::uint32_t cudaaligner_batches = 0,
      std::uint32_t cuda_aligner_band_width = 0);

  virtual std::vector<std::unique_ptr<biosoup::NucleicAcid>> Polish(
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences,
      bool drop_unpolished_sequences);

 protected:
  Polisher(
      std::shared_ptr<thread_pool::ThreadPool> thread_pool,
      std::uint64_t batch_size,
      double error_threshold,
      std::uint32_t window_len,
      bool trim_consensus,
      std::int8_t match,
      std::int8_t mismatch,
      std::int8_t gap);

  virtual void AllocateMemory(std::size_t /* step */) { /* dummy */ }

  virtual void DeallocateMemory(std::size_t /* step */) { /* dummy */ }

  virtual std::vector<Overlap> MapSequences(  // minimizers
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences);

  virtual void FindIntervals(  // global alignment
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences,
      std::vector<Overlap>* overlaps);

  virtual void GenerateConsensus();  // partial order alignment

  double e_;
  std::uint32_t w_;
  bool trim_;
  std::shared_ptr<thread_pool::ThreadPool> thread_pool_;
  std::uint64_t batch_size_;
  std::uint32_t mean_overlap_len_;
  std::vector<std::shared_ptr<Window>> windows_;
  std::vector<std::shared_ptr<spoa::AlignmentEngine>> alignment_engines_;
};

}  // namespace racon

#endif  // RACON_POLISHER_HPP_
