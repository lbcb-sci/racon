// Copyright (c) 2020 Robert Vaser

#include <algorithm>
#include <stdexcept>

#include "spoa/graph.hpp"

#include "window.hpp"

namespace racon {

Window::Window(
    std::uint64_t id,
    std::uint32_t rank,
    WindowType type,
    const char* backbone, std::uint32_t backbone_len,
    const char* quality, uint32_t quality_len)
    : id(id),
      rank(rank),
      type(type),
      status(false),
      consensus(),
      sequences(1, std::make_pair(backbone, backbone_len)),
      qualities(1, std::make_pair(quality, quality_len)),
      positions(1, std::make_pair(0, 0)) {
}

void Window::AddLayer(
    const char* sequence, std::uint32_t sequence_len,
    const char* quality, std::uint32_t quality_len,
    std::uint32_t begin,
    std::uint32_t end) {
  if (quality != nullptr && sequence_len != quality_len) {
    throw std::invalid_argument(
        "[racon::Window::AddLayer] error: unequal quality size");
  }
  if (begin >= end ||
      begin > sequences.front().second ||
      end > sequences.front().second) {
    throw std::invalid_argument(
        "[racon::Window::AddLayer] error: invalid positions");
  }

  sequences.emplace_back(sequence, sequence_len);
  qualities.emplace_back(quality, quality_len);
  positions.emplace_back(begin, end);
}

void Window::GenerateConsensus(
    std::shared_ptr<spoa::AlignmentEngine> alignment_engine,
    bool trim) {
  if (!consensus.empty()) {
    return;
  }
  if (sequences.size() < 3) {
    consensus = std::string(sequences.front().first, sequences.front().second);
    return;
  }

  auto graph = spoa::Graph{};
  graph.AddAlignment(
      spoa::Alignment(),
      sequences.front().first, sequences.front().second,
      qualities.front().first, qualities.front().second);

  std::vector<std::uint32_t> rank;
  rank.reserve(sequences.size());
  for (std::uint32_t i = 0; i < sequences.size(); ++i) {
    rank.emplace_back(i);
  }

  std::stable_sort(rank.begin() + 1, rank.end(),
      [&] (std::uint32_t lhs, std::uint32_t rhs) {
        return positions[lhs].first < positions[rhs].first;
      });

  std::uint32_t offset = 0.01 * sequences.front().second;
  for (std::uint32_t j = 1; j < sequences.size(); ++j) {
    std::uint32_t i = rank[j];

    spoa::Alignment alignment;
    if (positions[i].first < offset &&
        positions[i].second > sequences.front().second - offset) {
      alignment = alignment_engine->Align(
          sequences[i].first, sequences[i].second,
          graph);
    } else {
      std::vector<const spoa::Graph::Node*> mapping;
      auto subgraph = graph.Subgraph(
          positions[i].first,
          positions[i].second,
          &mapping);
      alignment = alignment_engine->Align(
          sequences[i].first, sequences[i].second,
          subgraph);
      subgraph.UpdateAlignment(mapping, &alignment);
    }

    if (qualities[i].first == nullptr) {
      graph.AddAlignment(
          alignment,
          sequences[i].first, sequences[i].second);
    } else {
      graph.AddAlignment(
        alignment,
        sequences[i].first, sequences[i].second,
        qualities[i].first, qualities[i].second);
    }
  }

  std::vector<std::uint32_t> coverages;
  consensus = graph.GenerateConsensus(&coverages);

  if (type == WindowType::kTGS && trim) {
    std::uint32_t average_coverage = (sequences.size() - 1) / 2;

    std::int32_t begin = 0, end = consensus.size() - 1;
    for (; begin < static_cast<std::int32_t>(consensus.size()); ++begin) {
      if (coverages[begin] >= average_coverage) {
        break;
      }
    }
    for (; end >= 0; --end) {
      if (coverages[end] >= average_coverage) {
        break;
      }
    }

    if (begin < end) {
      consensus = consensus.substr(begin, end - begin + 1);
    }
  }

  status = true;
  return;
}

}  // namespace racon
