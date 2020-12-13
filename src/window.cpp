// Copyright (c) 2020 Robert Vaser

#include "window.hpp"

#include <algorithm>
#include <stdexcept>

#include "spoa/graph.hpp"

namespace racon {

Window::Window(std::uint64_t id, std::uint32_t rank, WindowType type)
    : id(id),
      rank(rank),
      type(type),
      status(false),
      consensus(),
      sequences_ids(),
      sequences_intervals(),
      sequences(),
      positions() {
}

void Window::AddLayer(
    std::uint32_t sequence_id,
    std::uint32_t sequence_begin,
    std::uint32_t sequence_end,
    std::uint32_t window_begin,
    std::uint32_t window_end) {
  sequences_ids.emplace_back(sequence_id);
  sequences_intervals.emplace_back(sequence_begin, sequence_end);
  positions.emplace_back(window_begin, window_end);
}

void Window::Fill(
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences) {
  if (sequences_ids.empty()) {
    return;
  }
  this->sequences.emplace_back(  // backbone
      targets[sequences_ids[0]]->Inflate(
          sequences_intervals[0].first,
          sequences_intervals[0].second - sequences_intervals[0].first));

  for (std::uint32_t i = 1; i < sequences_ids.size(); ++i) {
    this->sequences.emplace_back(
        sequences[sequences_ids[i]]->Inflate(
            sequences_intervals[i].first,
            sequences_intervals[i].second - sequences_intervals[i].first));
  }

  std::vector<std::uint32_t>{}.swap(sequences_ids);
  std::vector<std::pair<std::uint32_t, std::uint32_t>>{}.swap(sequences_intervals);  // NOLINT
}

void Window::GenerateConsensus(
    std::shared_ptr<spoa::AlignmentEngine> alignment_engine,
    bool trim) {
  if (!consensus.empty()) {
    return;
  }
  if (sequences.size() < 3) {
    consensus = sequences.front();
    return;
  }

  auto graph = spoa::Graph{};
  graph.AddAlignment(spoa::Alignment(), sequences.front(), 0);

  std::vector<std::uint32_t> rank;
  rank.reserve(sequences.size());
  for (std::uint32_t i = 0; i < sequences.size(); ++i) {
    rank.emplace_back(i);
  }

  std::stable_sort(rank.begin() + 1, rank.end(),
      [&] (std::uint32_t lhs, std::uint32_t rhs) {
        return positions[lhs].first < positions[rhs].first;
      });

  std::uint32_t offset = 0.01 * sequences.front().size();
  for (std::uint32_t j = 1; j < sequences.size(); ++j) {
    std::uint32_t i = rank[j];

    spoa::Alignment alignment;
    if (positions[i].first < offset &&
        positions[i].second - 1 > sequences.front().size() - offset) {
      alignment = alignment_engine->Align(sequences[i], graph);
    } else {
      std::vector<const spoa::Graph::Node*> mapping;
      auto subgraph = graph.Subgraph(
          positions[i].first,
          positions[i].second - 1,
          &mapping);
      alignment = alignment_engine->Align(sequences[i], subgraph);
      subgraph.UpdateAlignment(mapping, &alignment);
    }

    graph.AddAlignment(alignment, sequences[i]);
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

  std::vector<std::string>{}.swap(sequences);
  std::vector<std::pair<std::uint32_t, std::uint32_t>>{}.swap(positions);

  return;
}

}  // namespace racon
