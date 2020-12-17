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
      double q,
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
    try {
      auto sparser = bioparser::Parser<biosoup::NucleicAcid>::Create<
          bioparser::FastaParser>(RACON_DATA_PATH + sequences_path);
      sequences = sparser->Parse(-1);
    } catch (const std::invalid_argument& exception) {
      auto sparser = bioparser::Parser<biosoup::NucleicAcid>::Create<
          bioparser::FastqParser>(RACON_DATA_PATH + sequences_path);
      sequences = sparser->Parse(-1);
    }

    auto rparser = bioparser::Parser<biosoup::NucleicAcid>::Create<
        bioparser::FastaParser>(RACON_DATA_PATH + std::string("sample_reference.fasta.gz"));  // NOLINT
    reference = std::move(rparser->Parse(-1).front());

    polisher = racon::Polisher::Create(
        std::make_shared<thread_pool::ThreadPool>(4),
        q,
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
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 500, 3, -5, -4);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1528, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishQuality) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 3, -5, -4);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1268, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishLargeWindow) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 1000, 3, -5, -4);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1512, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishQualityLargeWindow) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 1000, 3, -5, -4);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1260, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishEditDistance) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 500, 1, -1, -1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1520, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishQualityEditDistance) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 1, -1, -1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1292, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

#ifdef CUDA_ENABLED

TEST_F(RaconPolisherTest, PolishCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 500, 3, -5, -4, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1528
  EXPECT_EQ(1577, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishQualityCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 3, -5, -4, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1268
  EXPECT_EQ(1281, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishCudaPoaCudaAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 500, 3, -5, -4, 1, false, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1528
  EXPECT_EQ(1577, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishQualityCudaPoaCudaAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 3, -5, -4, 1, false, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1268
  EXPECT_EQ(1281, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishLargeWindowCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 1000, 3, -5, -4, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1512
  EXPECT_EQ(2786, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishQualityLargeWindowCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 1000, 3, -5, -4, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1260
  EXPECT_EQ(3934, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishLargeWindowCudaPoaCudaAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 1000, 3, -5, -4, 1, false, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1512
  EXPECT_EQ(2789, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishEditDistanceCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 500, 3, -1, -1, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1520
  EXPECT_EQ(1854, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishQualityEditDistanceCudaPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 3, -1, -1, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1292
  EXPECT_EQ(1453, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}

TEST_F(RaconPolisherTest, PolishEditDistanceCudaPoaCudaAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 500, 3, -1, -1, 1, false, 1);  // NOLINT

  auto polished = polisher->Polish(targets, sequences, true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1520
  EXPECT_EQ(1855, EditDistance(polished.front()->InflateData(), reference->InflateData()));  // NOLINT
}
#endif

}  // namespace test
}  // namespace racon
