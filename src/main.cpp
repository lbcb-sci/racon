#include <getopt.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "bioparser/fasta_parser.hpp"
#include "bioparser/fastq_parser.hpp"
#include "biosoup/timer.hpp"

#include "racon/polisher.hpp"

std::atomic<std::uint32_t> biosoup::Sequence::num_objects{0};

namespace {

static const char* racon_version = RACON_VERSION;

static struct option options[] = {
  {"include-unpolished", no_argument, nullptr, 'u'},
  {"window-length", required_argument, nullptr, 'w'},
  {"quality-threshold", required_argument, nullptr, 'q'},
  {"error-threshold", required_argument, nullptr, 'e'},
  {"no-trimming", no_argument, nullptr, 'T'},
  {"match", required_argument, nullptr, 'm'},
  {"mismatch", required_argument, nullptr, 'n'},
  {"gap", required_argument, nullptr, 'g'},
#ifdef CUDA_ENABLED
  {"cuda-poa-batches", optional_argument, nullptr, 'c'},
  {"cuda-banded-alignment", no_argument, nullptr, 'b'},
  {"cuda-aligner-batches", required_argument, nullptr, 'a'},
#endif
  {"threads", required_argument, nullptr, 't'},
  {"version", no_argument, nullptr, 'v'},
  {"help", no_argument, nullptr, 'h'},
  {nullptr, 0, nullptr, 0}
};

std::unique_ptr<bioparser::Parser<biosoup::Sequence>> CreateParser(
    const std::string& path) {

  auto is_suffix = [] (const std::string& src, const std::string& suffix) {
    return src.size() < suffix.size() ? false :
        src.compare(src.size() - suffix.size(), suffix.size(), suffix) == 0;
  };

  if (is_suffix(path, ".fasta") || is_suffix(path, ".fasta.gz") ||
      is_suffix(path, ".fna")   || is_suffix(path, ".fna.gz") ||
      is_suffix(path, ".fa")    || is_suffix(path, ".fa.gz")) {
    try {
      return bioparser::Parser<biosoup::Sequence>::Create<bioparser::FastaParser>(path);  // NOLINT
    } catch (const std::invalid_argument& exception) {
      std::cerr << exception.what() << std::endl;
      return nullptr;
    }
  }
  if (is_suffix(path, ".fastq")    || is_suffix(path, ".fq") ||
      is_suffix(path, ".fastq.gz") || is_suffix(path, ".fq.gz")) {
    try {
      return bioparser::Parser<biosoup::Sequence>::Create<bioparser::FastqParser>(path);  // NOLINT
    } catch (const std::invalid_argument& exception) {
      std::cerr << exception.what() << std::endl;
      return nullptr;
    }
  }

  std::cerr << "[racon::] error: file " << path
            << " has unsupported format extension (valid extensions: .fasta, "
            << ".fasta.gz, .fa, .fa.gz, .fastq, .fastq.gz, .fq, .fq.gz)!"
            << std::endl;
  return nullptr;
}

void Help() {
  std::cout <<
      "usage: racon [options ...] <target> <sequences>\n"
      "\n"
      "  #default output is stdout\n"
      "  <target>/<sequences>\n"
      "    input file in FASTA/FASTQ format (can be compressed with gzip)\n"
      "\n"
      "  options:\n"
      "    -u, --include-unpolished\n"
      "      output unpolished target sequences\n"
      "    -q, --quality-threshold <float>\n"
      "      default: 10.0\n"
      "      threshold for average base quality of windows used in POA\n"
      "    -e, --error-threshold <float>\n"
      "      default: 0.3\n"
      "      maximum allowed error rate used for filtering overlaps\n"
      "    -w, --window-length <int>\n"
      "      default: 500\n"
      "      size of window on which POA is performed\n"
      "    --no-trimming\n"
      "      disables consensus trimming at window ends\n"
      "    -m, --match <int>\n"
      "      default: 3\n"
      "      score for matching bases\n"
      "    -n, --mismatch <int>\n"
      "      default: -5\n"
      "      score for mismatching bases\n"
      "    -g, --gap <int>\n"
      "      default: -4\n"
      "      gap penalty (must be negative)\n"
#ifdef CUDA_ENABLED
      "    -c, --cuda-poa-batches <int>\n"
      "      default: 0\n"
      "      number of batches for CUDA accelerated polishing per GPU\n"
      "    -b, --cuda-banded-alignment\n"
      "      use banding approximation for alignment on GPU\n"
      "    -a, --cuda-aligner-batches <int>\n"
      "      default: 0\n"
      "      number of batches for CUDA accelerated alignment per GPU\n"
#endif
      "    -t, --threads <int>\n"
      "      default: 1\n"
      "      number of threads\n"
      "    --version\n"
      "      prints the version number\n"
      "    -h, --help\n"
      "      prints the usage\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> input_paths;

  double q = 10.0;
  double e = 0.3;
  std::uint32_t w = 500;
  bool trim = true;

  std::int8_t m = 3;
  std::int8_t n = -5;
  std::int8_t g = -4;

  bool drop_unpolished = true;
  std::uint32_t num_threads = 1;

  std::uint32_t cuda_poa_batches = 0;
  std::uint32_t cuda_aligner_batches = 0;
  bool cuda_banded_alignment = false;

  std::string optstring = "uq:e:w:m:n:g:t:h";
#ifdef CUDA_ENABLED
  optstring += "c:b:a:";
#endif

  int32_t argument;
  while ((argument = getopt_long(argc, argv, optstring.c_str(), options, nullptr)) != -1) {
    switch (argument) {
      case 'u': drop_unpolished = false; break;
      case 'q': q = atof(optarg); break;
      case 'e': e = atof(optarg); break;
      case 'w': w = atoi(optarg); break;
      case 'T': trim = false; break;
      case 'm': m = atoi(optarg); break;
      case 'x': n = atoi(optarg); break;
      case 'g': g = atoi(optarg); break;
#ifdef CUDA_ENABLED
      case 'c':
        //if option c encountered, cudapoa_batches initialized with a default value of 1.
        cuda_poa_batches = 1;
        // next text entry is not an option, assuming it's the arg for option 'c'
        if (optarg == NULL && argv[optind] != NULL
            && argv[optind][0] != '-') {
          cuda_poa_batches = atoi(argv[optind++]);
        }
        // optional argument provided in the ususal way
        if (optarg != NULL) {
          cuda_poa_batches = atoi(optarg);
        }
        break;
      case 'b':
        cuda_banded_alignment = true;
        break;
      case 'a':
        cuda_aligner_batches = atoi(optarg);
        break;
#endif
      case 't': num_threads = atoi(optarg); break;
      case 'v': std::cout << racon_version << std::endl; return 0;
      case 'h': Help(); return 0;
      default: return 1;
    }
  }

  if (argc == 1) {
    Help();
    return 0;
  }

  for (std::int32_t i = optind; i < argc; ++i) {
    input_paths.emplace_back(argv[i]);
  }

  if (input_paths.size() < 2) {
    std::cerr << "[racon::] error: missing input file(s)!" << std::endl;
    return 1;
  }

  biosoup::Timer timer{};
  timer.Start();

  auto tparser = CreateParser(input_paths[0]);
  if (tparser == nullptr) {
    return 1;
  }
  std::vector<std::unique_ptr<biosoup::Sequence>> targets;
  try {
    targets = tparser->Parse(-1);
  } catch (std::invalid_argument& exception) {
    std::cerr << exception.what() << std::endl;
    return 1;
  }

  std::cerr << "[racon::] loaded target sequences "
            << std::fixed << timer.Stop() << "s"
            << std::endl;

  timer.Start();

  biosoup::Sequence::num_objects = 0;
  auto sparser = CreateParser(input_paths[1]);
  if (sparser == nullptr) {
    return 1;
  }
  std::vector<std::unique_ptr<biosoup::Sequence>> sequences;
  try {
    sequences = sparser->Parse(-1);
  } catch (std::invalid_argument& exception) {
    std::cerr << exception.what() << std::endl;
    return 1;
  }

  std::cerr << "[racon::] loaded sequences "
            << std::fixed << timer.Stop() << "s"
            << std::endl;

  auto thread_pool = std::make_shared<thread_pool::ThreadPool>(num_threads);

  std::unique_ptr<racon::Polisher> polisher = nullptr;
  try {
    polisher = racon::Polisher::Create(q, e, w, trim, m, n, g, thread_pool,
        cuda_poa_batches, cuda_banded_alignment, cuda_aligner_batches);
  } catch (std::invalid_argument& exception) {
    std::cerr << exception.what() << std::endl;
    return 1;
  }
  polisher->Initialize(targets, sequences);

  auto polished = polisher->Polish(drop_unpolished);

  for (const auto& it : polished) {
    std::cout << ">" << it->name << std::endl
              << it->data << std::endl;
  }

  std::cerr << "[racon::] "
            << std::fixed << timer.elapsed_time() << "s"
            << std::endl;

  return 0;
}
