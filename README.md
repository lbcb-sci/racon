# Racon

[![Latest GitHub release](https://img.shields.io/github/release/lbcb-sci/racon.svg)](https://github.com/lbcb-sci/racon/releases/latest)
[![Build status for gcc/clang](https://travis-ci.org/lbcb-sci/racon.svg?branch=master)](https://travis-ci.org/lbcb-sci/racon)
[![Published in Genome Research](https://img.shields.io/badge/published%20in-Genome%20Research-blue.svg)](https://doi.org/10.1101/gr.214270.116)

Racon is a c++ consensus module for raw de novo DNA assembly of long uncorrected reads.

## Usage

To build racon run the following commands:
```bash
git clone --recursive https://github.com/lbcb-sci/racon.git racon
cd racon && mkdir build && cd build
cmake -Dracon_build_executable=ON -DCMAKE_BUILD_TYPE=Release .. && make
./bin/racon
```
which will display the following usage:
```bash
usage: racon [options ...] <target> <sequences>

  #default output is stdout
  <target>/<sequences>
    input file in FASTA/FASTQ format (can be compressed with gzip)

  options:
    -u, --include-unpolished
      output unpolished target sequences
    -q, --quality-threshold <float>
      default: 10.0
      threshold for average base quality of windows used in POA
    -e, --error-threshold <float>
      default: 0.3
      maximum allowed error rate used for filtering overlaps
    -w, --window-length <int>
      default: 500
      size of window on which POA is performed
    --no-trimming
      disables consensus trimming at window ends
    -m, --match <int>
      default: 3
      score for matching bases
    -n, --mismatch <int>
      default: -5
      score for mismatching bases
    -g, --gap <int>
      default: -4
      gap penalty (must be negative)
    -t, --threads <int>
      default: 1
      number of threads
    --version
      prints the version number
    -h, --help
      prints the usage

  only available when built with CUDA:
    -c, --cuda-poa-batches <int>
      default: 0
      number of batches for CUDA accelerated polishing per GPU
    -b, --cuda-banded-alignment
      use banding approximation for alignment on GPU
    -a, --cuda-aligner-batches <int>
      default: 0
      number of batches for CUDA accelerated alignment per GPU
```
If you would like to add racon as a library to your project via CMake, add the following:
```cmake
if (NOT TARGET racon)
  add_subdirectory(<path_to_submodules>/racon EXCLUDE_FROM_ALL)
endif ()
target_link_libraries(<your_exe> racon)
```

#### Dependencies
- gcc 4.8+ or clang 3.5+
- cmake 3.9+
- zlib (for binary only)

### CUDA Support
Racon makes use of [NVIDIA's ClaraGenomicsAnalysis SDK](https://github.com/clara-genomics/ClaraGenomicsAnalysis) for CUDA accelerated polishing and alignment.

To build racon with CUDA support, add `-Dracon_enable_cuda=ON` while running `cmake`. If CUDA support is unavailable, the `cmake` step will error out.
Note that the CUDA support flag does not produce a new binary target. Instead it augments the existing racon binary itself.

***Note***: Short read polishing with CUDA is still in development!

#### Dependencies
- gcc 5.0+
- cmake 3.10+
- CUDA 9.0+

## Unit tests

To build racon unit tests run the following commands:
```bash
git clone --recursive https://github.com/lbcb-sci/racon.git racon
cd racon && mkdir build && cd build
cmake -Dracon_build_tests=ON -DCMAKE_BUILD_TYPE=Release .. && make
./bin/racon_test
```

#### Dependencies
- gtest

## Acknowledgment

This work has been supported in part by Croatian Science Foundation under the project UIP-11-2013-7353. IS is supported in part by the Croatian Academy of Sciences and Arts under the project "Methods for alignment and assembly of DNA sequences using nanopore sequencing data". NN is supported by funding from A*STAR, Singapore.
