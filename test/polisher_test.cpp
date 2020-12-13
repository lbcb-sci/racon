// Copyright (c) 2020 Robert Vaser

#include "racon/polisher.hpp"

#include "bioparser/fasta_parser.hpp"
#include "bioparser/fastq_parser.hpp"
#include "edlib.h"  // NOLINT
#include "gtest/gtest.h"

std::atomic<std::uint32_t> biosoup::NucleicAcid::num_objects{0};

namespace racon {
namespace test {

class RaconPolisherTest: public ::testing::Test {
 public:
  void Setup(
      const std::string& targets_path,
      const std::string& sequences_path,
      std::uint64_t batch_size,
      double e,
      std::uint32_t w,
      std::int8_t m,
      std::int8_t n,
      std::int8_t g,
      std::uint32_t cudapoa_batches = 0,
      bool cuda_banded_alignment = false,
      std::uint32_t cudaaligner_batches = 0) {
    biosoup::NucleicAcid::num_objects = 0;
    auto tparser = bioparser::Parser<biosoup::NucleicAcid>::Create<
        bioparser::FastaParser>(RACON_DATA_PATH + targets_path);
    targets = tparser->Parse(-1);

    biosoup::NucleicAcid::num_objects = 0;
    auto sparser = bioparser::Parser<biosoup::NucleicAcid>::Create<
        bioparser::FastaParser>(RACON_DATA_PATH + sequences_path);
    sequences = sparser->Parse(-1);

    auto rparser = bioparser::Parser<biosoup::NucleicAcid>::Create<
        bioparser::FastaParser>(RACON_DATA_PATH + std::string("sample_reference.fasta.gz"));  // NOLINT
    reference = std::move(rparser->Parse(-1).front());

    polisher = racon::Polisher::Create(
        std::make_shared<thread_pool::ThreadPool>(4),
        batch_size,
        e,
        w,
        true,
        m,
        n,
        g,
        cudapoa_batches,
        cuda_banded_alignment,
        cudaaligner_batches);
  }

  static std::uint32_t EditDistance(
      const std::string& lhs,
      const std::string& rhs) {
    EdlibAlignResult result = edlibAlign(
        lhs.c_str(), lhs.size(),
        rhs.c_str(), rhs.size(),
        edlibDefaultAlignConfig());
    std::uint32_t edit_distance = result.editDistance;
    edlibFreeAlignResult(result);
    return edit_distance;
  }

  std::vector<std::unique_ptr<biosoup::NucleicAcid>> targets;
  std::vector<std::unique_ptr<biosoup::NucleicAcid>> sequences;
  std::unique_ptr<biosoup::NucleicAcid> reference;
  std::unique_ptr<racon::Polisher> polisher;
};

TEST_F(RaconPolisherTest, WindowLengthError) {
  try {
    Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 0, 0, 0, 0, 0, 0);
  } catch (const std::invalid_argument& exception) {
    EXPECT_STREQ(exception.what(),
        "[racon::Polisher::Create] error: invalid window length");
  }
}

TEST_F(RaconPolisherTest, Polish) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", -1, 0.3, 500, 3, -5, -4);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1528, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishBatch) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 1ULL << 19, 0.3, 500, 3, -5, -4);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1528, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishLargeWindow) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", -1, 0.3, 1000, 3, -5, -4);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1512, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishEditDistance) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", -1, 0.3, 500, 1, -1, -1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1520, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

#ifdef CUDA_ENABLED

TEST_F(RaconPolisherTest, PolishCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", -1, 0.3, 500, 3, -5, -4, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1528
  EXPECT_EQ(1577, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishBatchCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 1UL << 19, 0.3, 500, 3, -5, -4, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1528
  EXPECT_EQ(1577, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishCudaPoaCudaAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", -1, 0.3, 500, 3, -5, -4, 1, false, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1528
  EXPECT_EQ(1577, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishLargeWindowCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", -1, 0.3, 1000, 3, -5, -4, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1512
  EXPECT_EQ(2786, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishLargeWindowCudaPoaCudaAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", -1, 0.3, 1000, 3, -5, -4, 1, false, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1512
  EXPECT_EQ(2789, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishEditDistanceCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", -1, 0.3, 500, 3, -1, -1, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1520
  EXPECT_EQ(1854, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishEditDistanceCudaPoaCudaAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", -1, 0.3, 500, 3, -1, -1, 1, false, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1520
  EXPECT_EQ(1855, EditDistance(polished.front()->Inflate(), reference->Inflate()));  // NOLINT
}
#endif

}  // namespace test
}  // namespace racon
