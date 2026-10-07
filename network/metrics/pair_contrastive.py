"""Rank-local validation diagnostics using the contrastive loss truth rules."""

from fnmatch import fnmatchcase
from typing import Mapping

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor

from evenet.network.loss.pair_contrastive import (
    LEVELS,
    HierarchicalPairContrastiveLoss,
    build_contrastive_masks,
)
from evenet.network.metrics.pair_representation import PairMonitorResult


_GROUPS = {
    "same_event_positive": ("Same-event positive", "#3569A8", "-"),
    "cross_event_positive": ("Cross-event positive", "#4B9B8A", "--"),
    "endpoint_negative": ("Endpoint negative", "#C44E52", "-."),
    "random_negative": ("Random negative", "#D9772A", ":"),
}

_DEFAULT_SCALAR_METRICS = (
    "num_local_pairs",
    "embedding/*",
    "cosine/*/num_valid_anchors",
    "cosine/*/z/*_mean",
)


def embedding_statistics(features: Tensor) -> dict[str, float]:
    """Measure unstandardized variance and effective rank of centered covariance.

    Effective rank is exp(entropy(eigenvalues / sum(eigenvalues))). Constant
    embeddings have rank zero; fewer than two samples have undefined statistics.
    """
    if features.shape[0] < 2:
        return {"mean_variance": float("nan"), "effective_rank": float("nan")}
    centered = features.double() - features.double().mean(dim=0)
    variance = centered.square().mean().item()
    if variance == 0:
        return {"mean_variance": 0.0, "effective_rank": 0.0}
    energy = torch.linalg.svdvals(centered).square()
    weights = energy / energy.sum()
    weights = weights[weights > 0]
    rank = (-(weights * weights.log()).sum()).exp().item()
    return {"mean_variance": variance, "effective_rank": rank}


def cross_process_statistics(
    normalized: Tensor, events: Tensor, processes: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mean cosine and directed comparison counts for cross-event bonds.

    Inputs contain only bonds of one truth category at one level. Summed unit
    vectors give the exact pairwise mean without constructing a pair matrix.
    Diagonal cells exclude every within-event comparison, including self-pairs.
    """
    process_ids, inverse = torch.unique(processes, sorted=True, return_inverse=True)
    features = normalized.double()
    sums = features.new_zeros((process_ids.numel(), features.shape[1]))
    sums.index_add_(0, inverse, features)
    sizes = torch.bincount(inverse, minlength=process_ids.numel())
    totals = sums @ sums.T
    counts = sizes[:, None] * sizes[None, :]

    event_groups, event_inverse = torch.unique(
        torch.stack((inverse, events), dim=1), dim=0, return_inverse=True,
    )
    event_sums = features.new_zeros((event_groups.shape[0], features.shape[1]))
    event_sums.index_add_(0, event_inverse, features)
    event_sizes = torch.bincount(event_inverse)
    excluded_sums = features.new_zeros(process_ids.numel())
    excluded_sums.index_add_(0, event_groups[:, 0], event_sums.square().sum(dim=1))
    excluded_counts = torch.zeros_like(sizes)
    excluded_counts.index_add_(0, event_groups[:, 0], event_sizes.square())
    totals.diagonal().sub_(excluded_sums)
    counts.diagonal().sub_(excluded_counts)
    means = (totals / counts.clamp_min(1)).clamp(-1, 1)
    means[counts == 0] = float("nan")
    return process_ids, means, counts


class PairContrastiveMonitor:
    """Plot one rank's validation batch without changing loss sampling or RNG.

    Only histogram anchors and covariance/SVD samples are capped. Histogram
    anchors still see all local candidates. Heatmaps use all available bonds.
    Call on rank zero only; this monitor performs no distributed collectives.
    """

    def __init__(
        self,
        config: Mapping,
        criterion: HierarchicalPairContrastiveLoss,
        process_names: Mapping[int, str],
        category_names: Mapping[int, str],
    ) -> None:
        self.criterion = criterion
        self.process_names = process_names
        self.category_names = category_names
        self.every_n_epochs = int(config.get("every_n_epochs", 1))
        self.max_anchors = int(config.get("max_anchors_per_level", 128))
        self.chunk_size = int(config.get("anchor_chunk_size", 32))
        self.max_embeddings = int(config.get("max_embedding_samples", 2048))
        self.bins = int(config.get("histogram_bins", 40))
        self.seed = int(config.get("random_seed", 12345))
        self.scalar_metrics = config.get("scalar_metrics", _DEFAULT_SCALAR_METRICS)
        self.cross_process_categories = config.get("cross_process_categories", [])
        self.log_cross_process_table = bool(config.get("log_cross_process_table", False))
        for name in ("scalar_metrics", "cross_process_categories"):
            patterns = getattr(self, name)
            if isinstance(patterns, str) or not isinstance(patterns, (list, tuple)) or any(
                not isinstance(pattern, str) for pattern in patterns
            ):
                raise ValueError(f"PairContrastive monitor {name} must be a list of glob patterns")
        if min(self.every_n_epochs, self.max_anchors, self.chunk_size, self.bins) < 1:
            raise ValueError("PairContrastive monitor intervals and caps must be positive")
        if self.max_embeddings < 2:
            raise ValueError("PairContrastive monitor max_embedding_samples must be >= 2")

    @torch.no_grad()
    def __call__(
        self,
        pair_state: Tensor,
        z: Tensor,
        selected: dict[str, Tensor],
        process_ids: Tensor,
        symmetrize_pair: bool,
    ) -> PairMonitorResult:
        """Receive dense PL [B,N,N,D], projected z [P,Dz], and flattened truth.

        PL diagnostics use the same optional (i,j)/(j,i) average as the head.
        Features are detached and calculations run on CPU in bounded chunks.
        """
        batch, i, j = (selected[key] for key in ("batch", "i", "j"))
        metrics = {"pair_contrastive/num_local_pairs": float(batch.numel())}
        if batch.numel() == 0:
            return PairMonitorResult(metrics=self._select_metrics(metrics), rows=[], figures={})
        if z.ndim != 2 or z.shape[0] != batch.numel():
            raise ValueError("Contrastive monitor z and selected pairs must align")

        pl = pair_state.detach()[batch, i, j]
        if symmetrize_pair:
            pl = 0.5 * (pl + pair_state.detach()[batch, j, i])
        states = {"pl": pl.float().cpu(), "z": z.detach().float().cpu()}
        local = {
            key: selected[key].detach().cpu()
            for key in ("relations", "categories", "event", "i", "j", "pair_id")
        }
        processes = process_ids.reshape(-1)[batch].detach().long().cpu()
        generator = torch.Generator().manual_seed(self.seed)
        sample = torch.randperm(batch.numel(), generator=generator)[:self.max_embeddings]
        metrics["pair_contrastive/embedding/num_samples"] = float(sample.numel())
        for stage, features in states.items():
            metrics.update({
                f"pair_contrastive/embedding/{stage}/{name}": value
                for name, value in embedding_statistics(features[sample]).items()
            })
        normalized = {stage: F.normalize(x, dim=-1) for stage, x in states.items()}
        figures, rows = {}, []
        for name in self.criterion.enabled_levels:
            level = LEVELS[name]
            bonds = local["relations"][:, level] & local["categories"][:, level].ge(0)
            anchors = torch.nonzero(bonds, as_tuple=True)[0]
            anchors = anchors[torch.randperm(anchors.numel(), generator=generator)]
            anchors = anchors[:self.max_anchors]
            figures[f"pair_contrastive/cosine/{name}"] = self._cosine_figure(
                normalized, local, anchors, level, name, generator, metrics,
            )
            for category in local["categories"][bonds, level].unique().tolist():
                keep = bonds & local["categories"][:, level].eq(category)
                category_name = self.category_names.get(category, str(category))
                if not any(fnmatchcase(category_name, pattern) for pattern in self.cross_process_categories):
                    continue
                key = f"pair_contrastive/cross_process/{name}/{category_name}"
                figures[key] = self._process_figure(
                    normalized, local["event"], processes, keep,
                    name, category_name, rows, metrics,
                )
        return PairMonitorResult(metrics=self._select_metrics(metrics), rows=rows, figures=figures)

    def _select_metrics(self, metrics):
        """Filter scalar output before it reaches any logger or dashboard."""
        return {
            key: value for key, value in metrics.items()
            if any(fnmatchcase(key.removeprefix("pair_contrastive/"), pattern)
                   for pattern in self.scalar_metrics)
        }

    def _cosine_figure(
        self, states, local, anchors, level, name, generator, metrics,
    ):
        histograms = {
            stage: {group: torch.zeros(self.bins) for group in _GROUPS}
            for stage in states
        }
        totals = {stage: {group: 0.0 for group in _GROUPS} for stage in states}
        valid_count = 0
        for index in anchors.split(self.chunk_size):
            same_event, positive, endpoint, random_candidates = build_contrastive_masks(
                local, index, level,
            )
            random = self.criterion._sample_random_negatives(random_candidates, generator)
            valid = positive.any(dim=1) & (endpoint | random).any(dim=1)
            valid_count += int(valid.sum())
            groups = {
                "same_event_positive": positive & same_event,
                "cross_event_positive": positive & ~same_event,
                "endpoint_negative": endpoint,
                "random_negative": random,
            }
            for stage, features in states.items():
                similarities = (features[index] @ features.T).clamp(-1, 1)
                for group, mask in groups.items():
                    values = similarities[mask & valid[:, None]]
                    histograms[stage][group] += torch.histc(values, self.bins, -1, 1)
                    totals[stage][group] += values.double().sum().item()

        prefix = f"pair_contrastive/cosine/{name}"
        metrics[f"{prefix}/num_sampled_anchors"] = float(anchors.numel())
        metrics[f"{prefix}/num_valid_anchors"] = float(valid_count)
        figure, axes = plt.subplots(1, len(states), figsize=(10, 4.5), squeeze=False)
        edges = torch.linspace(-1, 1, self.bins + 1).numpy()
        for axis, stage in zip(axes[0], states):
            for group, (label, color, style) in _GROUPS.items():
                counts = histograms[stage][group]
                count = int(counts.sum())
                metrics[f"{prefix}/{stage}/{group}_count"] = float(count)
                if count:
                    metrics[f"{prefix}/{stage}/{group}_mean"] = totals[stage][group] / count
                    axis.stairs(
                        (counts / (count * 2 / self.bins)).numpy(), edges,
                        label=f"{label} (n={count:,})", color=color, linestyle=style,
                    )
            axis.set(xlim=(-1, 1), xlabel="Cosine similarity", ylabel="Density", title=stage.upper())
            if not axis.get_legend_handles_labels()[0]:
                axis.text(0.5, 0.5, "No valid anchor comparisons", ha="center", transform=axis.transAxes)
        figure.suptitle(f"{name}: rank 0 validation batch | {valid_count} valid sampled anchors")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, fontsize=8)
        figure.tight_layout(rect=(0, 0, 1, 0.82))
        return figure

    def _process_figure(
        self, states, events, processes, keep, level_name, category_name, rows, metrics,
    ):
        size = int(processes[keep].unique().numel())
        figure, axes = plt.subplots(
            1, len(states), figsize=(max(9, 1.1 * size + 5), max(4.6, 0.5 * size + 2)),
            squeeze=False, constrained_layout=True,
        )
        color_map = plt.get_cmap("coolwarm").with_extremes(bad="#DDDDDD")
        for axis, (stage, features) in zip(axes[0], states.items()):
            ids, means, counts = cross_process_statistics(features[keep], events[keep], processes[keep])
            names = [self.process_names.get(int(pid), str(int(pid))) for pid in ids]
            artist = axis.imshow(means.numpy(), vmin=-1, vmax=1, cmap=color_map)
            axis.set_xticks(range(size), names, rotation=45, ha="right", fontsize=8)
            axis.set_yticks(range(size), names, fontsize=8)
            axis.set(title=stage.upper(), xlabel="Candidate process", ylabel="Anchor process")
            figure.colorbar(artist, ax=axis, label="Mean cosine")
            for a, source in enumerate(names):
                for b, target in enumerate(names):
                    count = int(counts[a, b])
                    mean = float(means[a, b]) if count else None
                    label = f"{mean:.2f}\nn={count:,}" if count else "N/A\nn=0"
                    axis.text(b, a, label, ha="center", va="center", fontsize=max(5, 8 - size // 6))
                    if self.log_cross_process_table:
                        rows.append({
                            "level": level_name, "category": category_name, "stage": stage,
                            "anchor_process": source, "candidate_process": target,
                            "count": count, "mean_cosine": mean,
                        })
            off_diagonal = ~torch.eye(size, dtype=torch.bool)
            cross_count = int(counts[off_diagonal].sum())
            prefix = f"pair_contrastive/cross_process/{level_name}/{category_name}/{stage}"
            metrics[f"{prefix}/count"] = float(cross_count)
            if cross_count:
                metrics[f"{prefix}/mean"] = float(
                    (means.nan_to_num() * counts)[off_diagonal].sum() / cross_count
                )
        figure.suptitle(
            f"{level_name} / {category_name}: cross-event bonds, rank 0 batch\n"
            "Cells show directed comparison counts; within-event pairs excluded"
        )
        return figure
