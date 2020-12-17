// Copyright (c) 2020 Robert Vaser

#include "window.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "spoa/graph.hpp"

namespace racon {

Window::Window(
    std::uint64_t id,
    std::uint32_t rank,
    WindowType type,
    const std::string& backbone)
    : id(id),
      rank(rank),
      type(type),
      status(false),
      backbone(backbone),
      consensus(),
      sequences_ids(),
      sequences_intervals(),
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

void Window::GenerateConsensus(
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences,
    std::shared_ptr<spoa::AlignmentEngine> alignment_engine,
    bool trim) {
  if (!consensus.empty()) {
    return;
  }
  if (sequences_ids.size() < 2) {
    consensus = backbone;
    return;
  }

  auto graph = spoa::Graph{};
  graph.AddAlignment(spoa::Alignment(), backbone, 0);

  std::vector<std::uint32_t> rank(sequences_ids.size());
  std::iota(rank.begin(), rank.end(), 0);
  std::stable_sort(
      rank.begin(),
      rank.end(),
      [&] (std::uint32_t lhs, std::uint32_t rhs) {
        return positions[lhs].first < positions[rhs].first;
      });

  std::uint32_t offset = 0.01 * backbone.size();
  for (std::uint32_t j = 0; j < rank.size(); ++j) {
    std::uint32_t i = rank[j];

    std::string sequence = sequences[sequences_ids[i]]->InflateData(
        sequences_intervals[i].first,
        sequences_intervals[i].second - sequences_intervals[i].first);
    std::string quality = sequences[sequences_ids[i]]->InflateQuality(
        sequences_intervals[i].first,
        sequences_intervals[i].second - sequences_intervals[i].first);

    spoa::Alignment alignment;
    if (positions[i].first < offset &&
        positions[i].second - 1 > backbone.size() - offset) {
      alignment = alignment_engine->Align(sequence, graph);
    } else {
      std::vector<const spoa::Graph::Node*> mapping;
      auto subgraph = graph.Subgraph(
          positions[i].first,
          positions[i].second - 1,
          &mapping);
      alignment = alignment_engine->Align(sequence, subgraph);
      subgraph.UpdateAlignment(mapping, &alignment);
    }

    if (quality.empty()) {
      graph.AddAlignment(alignment, sequence);
    } else {
      graph.AddAlignment(alignment, sequence, quality);
    }
  }

  std::vector<std::uint32_t> coverages;
  consensus = graph.GenerateConsensus(&coverages);

  if (type == WindowType::kTGS && trim) {
    std::uint32_t average_coverage = rank.size() / 2;

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

  std::vector<std::uint32_t>{}.swap(sequences_ids);
  std::vector<std::pair<std::uint32_t, std::uint32_t>>{}.swap(sequences_intervals);  // NOLINT
  std::vector<std::pair<std::uint32_t, std::uint32_t>>{}.swap(positions);

  return;
}

}  // namespace racon
