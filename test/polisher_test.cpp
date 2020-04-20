/*!
 * @file racon_test.cpp
 *
 * @brief Racon unit test source file
 */

#include "racon/polisher.hpp"

#include "bioparser/fasta_parser.hpp"
#include "bioparser/fastq_parser.hpp"
#include "edlib.h"  // NOLINT
#include "gtest/gtest.h"

std::atomic<std::uint32_t> biosoup::Sequence::num_objects{0};

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
    biosoup::Sequence::num_objects = 0;
    auto tparser = bioparser::Parser<biosoup::Sequence>::Create<
        bioparser::FastaParser>(RACON_DATA_PATH + targets_path);
    targets = tparser->Parse(-1);

    biosoup::Sequence::num_objects = 0;
    try {
      auto sparser = bioparser::Parser<biosoup::Sequence>::Create<
          bioparser::FastqParser>(RACON_DATA_PATH + sequences_path);
      sequences = sparser->Parse(-1);
    } catch (const std::invalid_argument& exception) {
      auto sparser = bioparser::Parser<biosoup::Sequence>::Create<
          bioparser::FastaParser>(RACON_DATA_PATH + sequences_path);
      sequences = sparser->Parse(-1);
    }

    auto rparser = bioparser::Parser<biosoup::Sequence>::Create<
        bioparser::FastaParser>(RACON_DATA_PATH + std::string("sample_reference.fasta.gz"));  // NOLINT
    reference = std::move(rparser->Parse(-1).front());

    polisher = racon::Polisher::Create(
        q,
        e,
        w,
        true,
        m,
        n,
        g,
        std::make_shared<thread_pool::ThreadPool>(4),
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

  std::vector<std::unique_ptr<biosoup::Sequence>> targets;
  std::vector<std::unique_ptr<biosoup::Sequence>> sequences;
  std::unique_ptr<biosoup::Sequence> reference;
  std::unique_ptr<racon::Polisher> polisher;
};

TEST_F(RaconPolisherTest, WindowLengthError) {
  try {
    Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 0, 0, 0, 0, 0, 0);
  } catch (const std::invalid_argument& exception) {
    EXPECT_STREQ(exception.what(),
        "[racon::Polisher::Create] error: invalid window length");
  }
}

TEST_F(RaconPolisherTest, ConsensusWithQualities) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 3, -5, -4);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1291, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithoutQualities) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 500, 3, -5, -4);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1528, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithQualitiesLargerWindow) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 1000, 3, -5, -4);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1273, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithQualitiesEditDistance) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 1, -1, -1);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  EXPECT_EQ(1284, EditDistance(polished.front()->data, reference->data));
}

#ifdef CUDA_ENABLED
TEST_F(RaconPolisherTest, ConsensusWithQualitiesCUDAPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 3, -5, -4, 1);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1291
  EXPECT_EQ(1310, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithQualitiesCUDAPoaAndAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 3, -5, -4, 1, false, 1);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1291
  EXPECT_EQ(1315, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithoutQualitiesCUDAPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 500, 3, -5, -4, 1);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1528
  EXPECT_EQ(1577, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithoutQualitiesCUDAPoaAndAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fasta.gz", 10, 0.3, 500, 3, -5, -4, 1, false, 1);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1528
  EXPECT_EQ(1577, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithQualitiesLargerWindowCUDAPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 1000, 3, -5, -4, 1);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1273
  EXPECT_EQ(3403, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithQualitiesLargerWindowCUDAPoaAndAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 1000, 3, -5, -4, 1, false, 1);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1273
  EXPECT_EQ(3423, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithQualitiesEditDistanceCUDAPoa) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 3, -1, -1, 1);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1284
  EXPECT_EQ(1499, EditDistance(polished.front()->data, reference->data));
}

TEST_F(RaconPolisherTest, ConsensusWithQualitiesEditDistanceCUDAPoaAndAlignment) {
  Setup("sample_layout.fasta.gz", "sample_reads.fastq.gz", 10, 0.3, 500, 3, -1, -1, 1, false, 1);  // NOLINT

  polisher->Initialize(targets, sequences);
  auto polished = polisher->Polish(true);
  EXPECT_EQ(polished.size(), 1);

  polished.front()->ReverseAndComplement();
  // CPU 1284
  EXPECT_EQ(1493, EditDistance(polished.front()->data, reference->data));
}
#endif

}  // namespace test
}  // namespace racon
