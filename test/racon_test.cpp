/*!
 * @file racon_test.cpp
 *
 * @brief Racon unit test source file
 */

#include <algorithm>

#include "racon_test_config.h"

#include "thread_pool/thread_pool.hpp"
#include "bioparser/bioparser.hpp"
#include "ram/sequence.hpp"
#include "gtest/gtest.h"
#include "edlib.h"

#include "racon/polisher.hpp"

std::uint32_t calculateEditDistance(const std::string& query, const std::string& target) {

    EdlibAlignResult result = edlibAlign(query.c_str(), query.size(), target.c_str(),
        target.size(), edlibDefaultAlignConfig());

    std::uint32_t edit_distance = result.editDistance;
    edlibFreeAlignResult(result);

    return edit_distance;
}

void reverse_complement(const std::unique_ptr<ram::Sequence>& s) {
    for (auto& it: s->data) {
        switch (it) {
            case 'a': case 'A': it = 'T'; break;
            case 'c': case 'C': it = 'G'; break;
            case 'g': case 'G': it = 'C'; break;
            case 't': case 'T': it = 'A'; break;
            default: break;
        }
    }
    std::reverse(s->data.begin(), s->data.end());
}

class RaconPolishingTest: public ::testing::Test {
public:
    void SetUp(const std::string& target_path, const std::string& sequences_path,
        double q, double e, std::uint32_t w, std::int8_t m, std::int8_t n,
        std::int8_t g, std::uint32_t cudapoa_batches = 0,
        bool cuda_banded_alignment = false, std::uint32_t cudaaligner_batches = 0) {

        ram::Sequence::num_objects = 0;
        auto tparser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(target_path);
        tparser->parse(targets, -1);

        ram::Sequence::num_objects = 0;
        try {
            auto sparser = bioparser::createParser<bioparser::FastqParser, ram::Sequence>(sequences_path);
            sparser->parse(sequences, -1);
        } catch (const std::invalid_argument& exception) {
            auto sparser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(sequences_path);
            sparser->parse(sequences, -1);
        }

        polisher = racon::createPolisher(q, e, w, true, m, n, g,
            thread_pool::createThreadPool(4), cudapoa_batches,
            cuda_banded_alignment, cudaaligner_batches);
    }

    void TearDown() {}

    void initialize() {
        polisher->initialize(targets, sequences);
    }

    void polish(std::vector<std::unique_ptr<ram::Sequence>>& dst,
        bool drop_unpolished) {
        return polisher->polish(dst, drop_unpolished);
    }

    std::vector<std::unique_ptr<ram::Sequence>> targets;
    std::vector<std::unique_ptr<ram::Sequence>> sequences;
    std::unique_ptr<racon::Polisher> polisher;
};

TEST(RaconInitializeTest, WindowLengthError) {
    try {
        auto polished = racon::createPolisher(0, 0, 0, false, 0, 0, 0, nullptr,
            0, 0, 0);
    } catch (const std::invalid_argument& exception) {
        EXPECT_STREQ(exception.what(), "[racon::createPolisher] error: "
            "invalid window length!");
    }
}

TEST_F(RaconPolishingTest, ConsensusWithQualities) {
    SetUp(racon_test_data_path + "sample_layout.fasta.gz", racon_test_data_path +
        "sample_reads.fastq.gz", 10, 0.3, 500, 3, -5, -4);

    initialize();

    std::vector<std::unique_ptr<ram::Sequence>> polished;
    polish(polished, true);
    EXPECT_EQ(polished.size(), 1);

    reverse_complement(polished[0]);

    auto parser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(
        racon_test_data_path + "sample_reference.fasta.gz");
    parser->parse(polished, -1);
    EXPECT_EQ(polished.size(), 2);

    EXPECT_EQ(calculateEditDistance(polished[0]->data, polished[1]->data), 1311);
}

TEST_F(RaconPolishingTest, ConsensusWithoutQualities) {
    SetUp(racon_test_data_path + "sample_layout.fasta.gz", racon_test_data_path +
        "sample_reads.fasta.gz", 10, 0.3, 500, 3, -5, -4);

    initialize();

    std::vector<std::unique_ptr<ram::Sequence>> polished;
    polish(polished, true);
    EXPECT_EQ(polished.size(), 1);

    reverse_complement(polished[0]);

    auto parser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(
        racon_test_data_path + "sample_reference.fasta.gz");
    parser->parse(polished, -1);
    EXPECT_EQ(polished.size(), 2);

    EXPECT_EQ(calculateEditDistance(polished[0]->data, polished[1]->data), 1579);
}

TEST_F(RaconPolishingTest, ConsensusWithQualitiesLargerWindow) {
    SetUp(racon_test_data_path + "sample_layout.fasta.gz", racon_test_data_path +
        "sample_reads.fastq.gz", 10, 0.3, 1000, 3, -5, -4);

    initialize();

    std::vector<std::unique_ptr<ram::Sequence>> polished;
    polish(polished, true);
    EXPECT_EQ(polished.size(), 1);

    reverse_complement(polished[0]);

    auto parser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(
        racon_test_data_path + "sample_reference.fasta.gz");
    parser->parse(polished, -1);
    EXPECT_EQ(polished.size(), 2);

    EXPECT_EQ(calculateEditDistance(polished[0]->data, polished[1]->data), 1290);
}

TEST_F(RaconPolishingTest, ConsensusWithQualitiesEditDistance) {
    SetUp(racon_test_data_path + "sample_layout.fasta.gz", racon_test_data_path +
        "sample_reads.fastq.gz", 10, 0.3, 500, 1, -1, -1);

    initialize();

    std::vector<std::unique_ptr<ram::Sequence>> polished;
    polish(polished, true);
    EXPECT_EQ(polished.size(), 1);

    reverse_complement(polished[0]);

    auto parser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(
        racon_test_data_path + "sample_reference.fasta.gz");
    parser->parse(polished, -1);
    EXPECT_EQ(polished.size(), 2);

    EXPECT_EQ(calculateEditDistance(polished[0]->data, polished[1]->data), 1327);
}

#ifdef CUDA_ENABLED
TEST_F(RaconPolishingTest, ConsensusWithQualitiesCUDA) {
    SetUp(racon_test_data_path + "sample_layout.fasta.gz", racon_test_data_path +
        "sample_reads.fastq.gz", 10, 0.3, 500, 3, -5, -4, 1);

    initialize();

    std::vector<std::unique_ptr<ram::Sequence>> polished;
    polish(polished, true);
    EXPECT_EQ(polished.size(), 1);

    reverse_complement(polished[0]);

    auto parser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(
        racon_test_data_path + "sample_reference.fasta.gz");
    parser->parse(polished, -1);
    EXPECT_EQ(polished.size(), 2);

    EXPECT_EQ(calculateEditDistance(polished[0]->data, polished[1]->data), 1341); // CPU 1311
}

TEST_F(RaconPolishingTest, ConsensusWithoutQualitiesCUDA) {
    SetUp(racon_test_data_path + "sample_layout.fasta.gz", racon_test_data_path +
        "sample_reads.fasta.gz", 10, 0.3, 500, 3, -5, -4, 1);

    initialize();

    std::vector<std::unique_ptr<ram::Sequence>> polished;
    polish(polished, true);
    EXPECT_EQ(polished.size(), 1);

    reverse_complement(polished[0]);

    auto parser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(
        racon_test_data_path + "sample_reference.fasta.gz");
    parser->parse(polished, -1);
    EXPECT_EQ(polished.size(), 2);

    EXPECT_EQ(calculateEditDistance(polished[0]->data, polished[1]->data), 1574); // CPU 1579
}

TEST_F(RaconPolishingTest, ConsensusWithQualitiesLargerWindowCUDA) {
    SetUp(racon_test_data_path + "sample_layout.fasta.gz", racon_test_data_path +
        "sample_reads.fastq.gz", 10, 0.3, 1000, 3, -5, -4, 1);

    initialize();

    std::vector<std::unique_ptr<ram::Sequence>> polished;
    polish(polished, true);
    EXPECT_EQ(polished.size(), 1);

    reverse_complement(polished[0]);

    auto parser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(
        racon_test_data_path + "sample_reference.fasta.gz");
    parser->parse(polished, -1);
    EXPECT_EQ(polished.size(), 2);

    EXPECT_EQ(calculateEditDistance(polished[0]->data, polished[1]->data), 5303); // CPU 1290
}

TEST_F(RaconPolishingTest, ConsensusWithQualitiesEditDistanceCUDA) {
    SetUp(racon_test_data_path + "sample_layout.fasta.gz", racon_test_data_path +
        "sample_reads.fastq.gz", 10, 0.3, 500, 3, -1, -1, 1);

    initialize();

    std::vector<std::unique_ptr<ram::Sequence>> polished;
    polish(polished, true);
    EXPECT_EQ(polished.size(), 1);

    reverse_complement(polished[0]);

    auto parser = bioparser::createParser<bioparser::FastaParser, ram::Sequence>(
        racon_test_data_path + "sample_reference.fasta.gz");
    parser->parse(polished, -1);
    EXPECT_EQ(polished.size(), 2);

    EXPECT_EQ(calculateEditDistance(polished[0]->data, polished[1]->data), 1444); // CPU 1327
}
#endif
