"""Lightweight monitoring for physics pair representations.

The monitor is intentionally stateless. Run it on an occasional, fixed
validation batch instead of accumulating tensors throughout an epoch.
"""

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from torch import Tensor


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
_PUBLICATION_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}
_FIGURE_DPI = 200

_GROUP_COLORS = (
    "#3569A8",
    "#D9772A",
    "#4B9B8A",
    "#8C6BB1",
    "#C44E52",
    "#7A9E3A",
    "#A66A4F",
    "#4C93B8",
)
_SPECIAL_STYLES = {
    "cross_segment": ("#555555", "^"),
    "one_unassigned": ("#8A8A8A", "x"),
    "both_unassigned": ("#B8B8B8", "s"),
}


@dataclass(frozen=True)
class PairGroupLabels:
    """Pair-group labels derived from truth segmentation instances."""

    labels: Tensor
    names: dict[int, str]


@dataclass(frozen=True)
class PairMonitorResult:
    """Scalar metrics and figures; nonzero DDP ranks return empty data."""

    metrics: dict[str, float]
    rows: list[dict[str, object]]
    figures: dict[str, Figure]


def build_pair_group_labels(
    target_masks: Tensor,
    target_classes: Tensor,
    pair_mask: Tensor,
    segment_class_names: Optional[Mapping[int, str]] = None,
    num_segment_classes: Optional[int] = None,
) -> PairGroupLabels:
    """Assign valid off-diagonal pairs to physics-motivated groups.

    A pair sharing a non-null truth segment is labelled by that segment class.
    If nested segments contain the same pair, the smallest segment is used as
    the most specific assignment. Remaining pairs are split into cross-segment,
    one-unassigned, and both-unassigned groups.

    Args:
        target_masks: Truth instance masks with shape ``[B, Q, N]``.
        target_classes: Integer classes ``[B, Q]`` or one-hot classes
            ``[B, Q, C]``. Class zero is the null/padding class.
        pair_mask: Valid pair mask with shape ``[B, N, N]``.
        segment_class_names: Optional mapping from non-null class IDs to names.
        num_segment_classes: Total class count including the null class. Only
            needed for integer targets when different ranks may observe
            different subsets of classes.
    """

    if target_masks.ndim != 3:
        raise ValueError("target_masks must have shape [B, Q, N]")
    if pair_mask.ndim != 3:
        raise ValueError("pair_mask must have shape [B, N, N]")

    batch_size, _, num_objects = target_masks.shape
    if pair_mask.shape != (batch_size, num_objects, num_objects):
        raise ValueError("pair_mask and target_masks describe different objects")

    if target_classes.ndim == 3:
        class_ids = target_classes.argmax(dim=-1)
        num_classes = target_classes.shape[-1]
        if num_segment_classes is not None and num_segment_classes != num_classes:
            raise ValueError("num_segment_classes disagrees with one-hot targets")
    elif target_classes.ndim == 2:
        class_ids = target_classes.long()
        num_classes = num_segment_classes or (
            int(class_ids.max().item()) + 1 if class_ids.numel() else 1
        )
        if class_ids.numel() and int(class_ids.max()) >= num_classes:
            raise ValueError("target class ID exceeds num_segment_classes")
    else:
        raise ValueError("target_classes must have shape [B, Q] or [B, Q, C]")

    if class_ids.shape != target_masks.shape[:2]:
        raise ValueError("target_classes and target_masks have incompatible shapes")

    active_masks = target_masks.bool() & class_ids.gt(0).unsqueeze(-1)
    object_is_assigned = active_masks.any(dim=1)

    shared_instances = active_masks.unsqueeze(-1) & active_masks.unsqueeze(-2)
    has_shared_instance = shared_instances.any(dim=1)

    instance_sizes = active_masks.sum(dim=-1).to(torch.float32)
    shared_sizes = torch.where(
        shared_instances,
        instance_sizes.unsqueeze(-1).unsqueeze(-1),
        torch.full((), torch.inf, device=target_masks.device),
    )
    most_specific_instance = shared_sizes.argmin(dim=1)
    shared_class = class_ids.gather(
        1, most_specific_instance.flatten(start_dim=1)
    ).reshape(batch_size, num_objects, num_objects)

    cross_segment_id = num_classes
    one_unassigned_id = num_classes + 1
    both_unassigned_id = num_classes + 2

    labels = torch.full_like(pair_mask, -1, dtype=torch.long)
    labels[has_shared_instance] = shared_class[has_shared_instance]

    assigned_i = object_is_assigned.unsqueeze(-1)
    assigned_j = object_is_assigned.unsqueeze(-2)
    labels[assigned_i & assigned_j & ~has_shared_instance] = cross_segment_id
    labels[assigned_i ^ assigned_j] = one_unassigned_id
    labels[~assigned_i & ~assigned_j] = both_unassigned_id

    diagonal = torch.eye(num_objects, dtype=torch.bool, device=pair_mask.device)
    valid_pairs = pair_mask.bool() & ~diagonal.unsqueeze(0)
    labels[~valid_pairs] = -1

    class_names = dict(segment_class_names or {})
    names = {
        class_id: class_names.get(class_id, f"segment_{class_id}")
        for class_id in range(1, num_classes)
    }
    names.update(
        {
            cross_segment_id: "cross_segment",
            one_unassigned_id: "one_unassigned",
            both_unassigned_id: "both_unassigned",
        }
    )
    return PairGroupLabels(labels=labels, names=names)


def centroid_separation_score(features: Tensor, labels: Tensor) -> float:
    """Return balanced between-group scatter divided by total scatter.

    Features are standardized per dimension first, so raw, P0, and PL scores
    remain comparable despite different units and latent dimensions. Each group
    receives equal weight. The result lies in ``[0, 1]``; higher means more
    compact and more separated group centroids.
    """

    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("features and labels must have shapes [M, D] and [M]")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must contain the same number of pairs")

    usable_groups = [
        group
        for group in torch.unique(labels).tolist()
        if int(labels.eq(group).sum()) >= 2
    ]
    if len(usable_groups) < 2:
        return float("nan")

    keep = torch.zeros_like(labels, dtype=torch.bool)
    for group in usable_groups:
        keep |= labels.eq(group)
    standardized = _standardize(features[keep].float())
    kept_labels = labels[keep]

    centroids = torch.stack(
        [standardized[kept_labels.eq(group)].mean(dim=0) for group in usable_groups]
    )
    balanced_center = centroids.mean(dim=0)
    between_scatter = (centroids - balanced_center).square().mean()

    within_scatter = torch.stack(
        [
            (
                standardized[kept_labels.eq(group)] - centroids[index]
            ).square().mean()
            for index, group in enumerate(usable_groups)
        ]
    ).mean()
    total_scatter = between_scatter + within_scatter
    if total_scatter <= torch.finfo(total_scatter.dtype).eps:
        return 0.0
    return float((between_scatter / total_scatter).clamp(0.0, 1.0).item())


class PairRepresentationMonitor:
    """Analyze capped raw, P0, and PL states, synchronized across DDP ranks."""

    def __init__(
        self,
        max_pairs_per_process_group: int = 256,
        max_pairs_per_event_group: int = 32,
        random_seed: int = 12345,
        include_all_processes: bool = True,
        plot_max_pairs_per_group: int = 128,
        sync_distributed: bool = True,
        process_plots: tuple[str, ...] | list[str] = (),
        plot_stages: tuple[str, ...] | list[str] = ("raw", "pl", "pl_minus_p0"),
    ) -> None:
        if (
            max_pairs_per_process_group < 1
            or max_pairs_per_event_group < 1
            or plot_max_pairs_per_group < 1
        ):
            raise ValueError("pair caps must be positive")
        self.max_pairs_per_process_group = max_pairs_per_process_group
        self.max_pairs_per_event_group = max_pairs_per_event_group
        self.random_seed = random_seed
        self.include_all_processes = include_all_processes
        self.plot_max_pairs_per_group = plot_max_pairs_per_group
        self.sync_distributed = sync_distributed
        if not isinstance(process_plots, (list, tuple)) or any(
            not isinstance(name, str) for name in process_plots
        ):
            raise ValueError("process_plots must be a list of process-name glob patterns")
        if (
            not isinstance(plot_stages, (list, tuple)) or not plot_stages
            or any(stage not in ("raw", "p0", "pl", "pl_minus_p0") for stage in plot_stages)
            or len(set(plot_stages)) != len(plot_stages)
        ):
            raise ValueError("plot_stages must be a nonempty, unique list of raw, p0, pl, pl_minus_p0")
        self.process_plots = tuple(process_plots)
        self.plot_stages = tuple(plot_stages)

    @torch.no_grad()
    def __call__(
        self,
        raw: Tensor,
        p0: Tensor,
        pl: Tensor,
        pair_mask: Tensor,
        target_masks: Tensor,
        target_classes: Tensor,
        process_ids: Tensor,
        process_names: Optional[Mapping[int, str]] = None,
        segment_class_names: Optional[Mapping[int, str]] = None,
        num_segment_classes: Optional[int] = None,
    ) -> PairMonitorResult:
        """Compute global scores and PCA; every DDP rank must call this method."""

        _validate_pair_states(raw, p0, pl, pair_mask)
        if p0.shape[-1] != pl.shape[-1]:
            raise ValueError("P0 and PL must have the same latent dimension")
        if (
            target_classes.ndim == 2
            and num_segment_classes is None
            and self.sync_distributed
            and dist.is_available()
            and dist.is_initialized()
        ):
            raise ValueError(
                "num_segment_classes is required for integer targets in distributed mode"
            )

        groups = build_pair_group_labels(
            target_masks=target_masks,
            target_classes=target_classes,
            pair_mask=pair_mask,
            segment_class_names=segment_class_names,
            num_segment_classes=num_segment_classes,
        )
        process_ids = process_ids.reshape(-1).long()
        if process_ids.shape[0] != raw.shape[0]:
            raise ValueError("process_ids must contain one ID per event")

        flat_labels = groups.labels.reshape(-1).cpu()
        flat_processes = process_ids[:, None, None].expand_as(groups.labels).reshape(-1).cpu()
        flat_events = (
            torch.arange(raw.shape[0])[:, None, None]
            .expand_as(groups.labels)
            .reshape(-1)
        )
        selected = self._sample_indices(flat_labels, flat_processes, flat_events)
        selected_indices = (
            torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)
        )

        selected_states = {
            "raw": _select_pairs(raw, selected_indices),
            "p0": _select_pairs(p0, selected_indices),
            "pl": _select_pairs(pl, selected_indices),
        }
        selected_labels = flat_labels[selected_indices]
        selected_processes = flat_processes[selected_indices]
        selected_states, selected_labels, selected_processes, is_global_zero = (
            self._reduce_across_ranks(
                selected_states,
                selected_labels,
                selected_processes,
            )
        )
        if not is_global_zero:
            return PairMonitorResult(metrics={}, rows=[], figures={})
        if selected_labels.numel() == 0:
            raise ValueError("no valid labelled pairs are available for monitoring")

        present_processes = sorted(torch.unique(selected_processes).tolist())
        name_map = dict(process_names or {})
        process_scopes: list[tuple[str, Optional[int]]] = []
        if self.include_all_processes:
            process_scopes.append(("all", None))
        process_scopes.extend(
            (name_map.get(int(process_id), f"process_{int(process_id)}"), int(process_id))
            for process_id in present_processes
            if any(fnmatchcase(name_map.get(int(process_id), f"process_{int(process_id)}"), pattern)
                   for pattern in self.process_plots)
        )

        metrics: dict[str, float] = {}
        if self.include_all_processes:
            # Pool the selected pairs before computing scores, not process-wise scores.
            scores = {
                stage: centroid_separation_score(state, selected_labels)
                for stage, state in selected_states.items()
            }
            metrics = {
                "pair_monitor/separation/all/raw": scores["raw"],
                "pair_monitor/separation/all/pl": scores["pl"],
                "pair_monitor/separation_gain/all/pl_minus_p0": scores["pl"] - scores["p0"],
                "pair_monitor/separation_gain/all/pl_minus_raw": scores["pl"] - scores["raw"],
                "pair_monitor/num_pairs": float(selected_labels.numel()),
                "pair_monitor/num_groups": float(selected_labels.unique().numel()),
            }

        figures = _plot_publication_figures(
            states=selected_states,
            labels=selected_labels,
            processes=selected_processes,
            process_scopes=process_scopes,
            group_names=groups.names,
            plot_stages=self.plot_stages,
            plot_max_pairs_per_group=self.plot_max_pairs_per_group,
            random_seed=self.random_seed,
        )
        return PairMonitorResult(metrics=metrics, rows=[], figures=figures)

    def _reduce_across_ranks(
        self,
        states: dict[str, Tensor],
        labels: Tensor,
        processes: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, bool]:
        if (
            not self.sync_distributed
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return states, labels, processes, True

        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (states, labels, processes))
        if dist.get_rank() != 0:
            return states, labels, processes, False

        states = {
            stage: torch.cat([payload[0][stage] for payload in gathered])
            for stage in states
        }
        labels = torch.cat([payload[1] for payload in gathered])
        processes = torch.cat([payload[2] for payload in gathered])
        keep = self._cap_process_groups(labels, processes)
        states = {stage: state[keep] for stage, state in states.items()}
        return states, labels[keep], processes[keep], True

    def _cap_process_groups(self, labels: Tensor, processes: Tensor) -> Tensor:
        generator = torch.Generator().manual_seed(self.random_seed)
        selected = []
        for process_id in torch.unique(processes).tolist():
            process_mask = processes.eq(process_id)
            for group_id in torch.unique(labels[process_mask]).tolist():
                candidates = torch.nonzero(
                    process_mask & labels.eq(group_id), as_tuple=False
                ).flatten()
                selected.append(
                    _random_cap(
                        candidates, self.max_pairs_per_process_group, generator
                    )
                )
        return torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)

    def _sample_indices(
        self,
        labels: Tensor,
        processes: Tensor,
        events: Tensor,
    ) -> list[Tensor]:
        generator = torch.Generator().manual_seed(self.random_seed)
        selected: list[Tensor] = []
        for process_id in torch.unique(processes[labels.ge(0)]).tolist():
            process_mask = processes.eq(process_id) & labels.ge(0)
            for group_id in torch.unique(labels[process_mask]).tolist():
                group_mask = process_mask & labels.eq(group_id)
                event_samples: list[Tensor] = []
                for event_id in torch.unique(events[group_mask]).tolist():
                    candidates = torch.nonzero(
                        group_mask & events.eq(event_id), as_tuple=False
                    ).flatten()
                    event_samples.append(
                        _random_cap(
                            candidates, self.max_pairs_per_event_group, generator
                        )
                    )
                candidates = torch.cat(event_samples)
                selected.append(
                    _random_cap(
                        candidates, self.max_pairs_per_process_group, generator
                    )
                )
        return selected


def _validate_pair_states(raw: Tensor, p0: Tensor, pl: Tensor, pair_mask: Tensor) -> None:
    for name, state in {"raw": raw, "p0": p0, "pl": pl}.items():
        if state.ndim != 4:
            raise ValueError(f"{name} must have shape [B, N, N, D]")
        if state.shape[:3] != pair_mask.shape:
            raise ValueError(f"{name} and pair_mask have incompatible shapes")


def _random_cap(candidates: Tensor, cap: int, generator: torch.Generator) -> Tensor:
    if candidates.numel() <= cap:
        return candidates
    order = torch.randperm(candidates.numel(), generator=generator)[:cap]
    return candidates[order]


def _select_pairs(state: Tensor, indices: Tensor) -> Tensor:
    flat_state = state.detach().reshape(-1, state.shape[-1])
    selected = flat_state.index_select(0, indices.to(flat_state.device))
    return selected.float().cpu()


def _standardize(features: Tensor) -> Tensor:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    varying = scale > torch.finfo(features.dtype).eps
    if not varying.any():
        return torch.zeros((features.shape[0], 1), dtype=features.dtype)
    return (features[:, varying] - mean[varying]) / scale[varying]


@dataclass(frozen=True)
class _PCAProjection:
    mean: Tensor
    scale: Tensor
    varying: Tensor
    components: Tensor
    explained_variance: tuple[float, float]

    def transform(self, features: Tensor) -> Tensor:
        standardized = (
            features[:, self.varying] - self.mean[self.varying]
        ) / self.scale[self.varying]
        coordinates = standardized @ self.components.T
        if coordinates.shape[1] == 1:
            coordinates = torch.cat(
                [coordinates, torch.zeros_like(coordinates)], dim=1
            )
        return coordinates


def _fit_pca(features: Tensor) -> _PCAProjection:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    varying = scale > torch.finfo(features.dtype).eps
    if not varying.any():
        varying = torch.zeros_like(scale, dtype=torch.bool)
        varying[0] = True
        scale = scale.clone()
        scale[0] = 1.0

    standardized = (features[:, varying] - mean[varying]) / scale[varying]
    _, singular_values, right_vectors = torch.linalg.svd(
        standardized, full_matrices=False
    )
    num_components = min(2, right_vectors.shape[0])
    components = right_vectors[:num_components]
    variance = singular_values.square()
    explained = variance[:num_components] / variance.sum().clamp_min(
        torch.finfo(variance.dtype).eps
    )
    explained_values = explained.tolist() + [0.0] * (2 - num_components)
    return _PCAProjection(
        mean=mean,
        scale=scale,
        varying=varying,
        components=components,
        explained_variance=(explained_values[0], explained_values[1]),
    )


def _plot_publication_figures(
    states: Mapping[str, Tensor],
    labels: Tensor,
    processes: Tensor,
    process_scopes: list[tuple[str, Optional[int]]],
    group_names: Mapping[int, str],
    plot_stages: tuple[str, ...],
    plot_max_pairs_per_group: int,
    random_seed: int,
) -> dict[str, Figure]:
    """One pooled overview, plus only requested process update distributions."""
    figures: dict[str, Figure] = {}
    generator = torch.Generator().manual_seed(random_seed)
    with plt.rc_context(_PUBLICATION_STYLE):
        for process_name, process_id in process_scopes:
            scope = (
                torch.ones_like(processes, dtype=torch.bool)
                if process_id is None
                else processes.eq(process_id)
            )
            plot_indices = _balanced_plot_indices(
                labels, scope, plot_max_pairs_per_group, generator,
            )
            stages = plot_stages if process_id is None else ("pl_minus_p0",)
            plot_states = {
                stage: (states["pl"][plot_indices] - states["p0"][plot_indices])
                if stage == "pl_minus_p0" else states[stage][plot_indices]
                for stage in stages
            }
            projections = {
                stage: _fit_pca(features) for stage, features in plot_states.items()
            }
            coordinates = {
                stage: projections[stage].transform(features)
                for stage, features in plot_states.items()
            }
            # Explained variance stays in panel titles instead of separate metrics.
            figures[f"pca/{_safe_key(process_name)}"] = _plot_process_figure(
                process_name=process_name,
                coordinates=coordinates,
                labels=labels[plot_indices],
                group_names=group_names,
                projections=projections,
            )
    return figures


def _balanced_plot_indices(
    labels: Tensor,
    scope: Tensor,
    cap: int,
    generator: torch.Generator,
) -> Tensor:
    """Select an equal visual sample per truth group without changing metrics."""

    groups = torch.unique(labels[scope]).tolist()
    if not groups:
        return torch.empty(0, dtype=torch.long)
    target = min(
        cap,
        min(int((scope & labels.eq(group)).sum()) for group in groups),
    )
    selected = []
    for group in groups:
        candidates = torch.nonzero(scope & labels.eq(group), as_tuple=False).flatten()
        selected.append(_random_cap(candidates, target, generator))
    return torch.cat(selected)


def _plot_process_figure(
    process_name: str,
    coordinates: Mapping[str, Tensor],
    labels: Tensor,
    group_names: Mapping[int, str],
    projections: Mapping[str, _PCAProjection],
) -> Figure:
    stages = tuple(coordinates)
    figure = plt.figure(figsize=(3.6 * len(stages) + 0.4, 3.8), dpi=_FIGURE_DPI, facecolor="white")
    grid = figure.add_gridspec(
        1,
        len(stages),
        left=0.075 if len(stages) > 1 else 0.16,
        right=0.985,
        top=0.80,
        bottom=0.18,
        wspace=0.42,
    )
    group_styles = _group_styles(labels, group_names)
    stage_titles = {
        "raw": "Raw pair features",
        "p0": "Initial latent P0",
        "pl": "Updated latent PL",
        "pl_minus_p0": "Latent update PL - P0",
    }
    panel_letters = "abcd"

    for column, stage in enumerate(stages):
        explained = sum(projections[stage].explained_variance)
        _draw_pca_distribution(
            figure=figure,
            slot=grid[column],
            coordinates=coordinates[stage],
            labels=labels,
            group_names=group_names,
            group_styles=group_styles,
            title=f"{stage_titles[stage]}\nPC1 + PC2 = {explained:.1%}",
            panel_letter=panel_letters[column],
        )

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markersize=4,
            markerfacecolor=color if marker != "x" else "none",
            markeredgecolor=color,
            markeredgewidth=0.7,
            label=group_names.get(group, f"group_{group}"),
        )
        for group, (color, marker) in group_styles.items()
    ]
    if handles:
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=min(6 if len(stages) > 1 else 3, len(handles)),
            columnspacing=1.0,
            handletextpad=0.35,
            frameon=False,
            fontsize=5.8,
        )
    figure.suptitle(
        f"Pair representations | {process_name}",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    return figure


def _draw_pca_distribution(
    figure: Figure,
    slot,
    coordinates: Tensor,
    labels: Tensor,
    group_names: Mapping[int, str],
    group_styles: Mapping[int, tuple[str, str]],
    title: str,
    panel_letter: str,
) -> None:
    nested = slot.subgridspec(
        2,
        2,
        width_ratios=(4.0, 1.0),
        height_ratios=(1.0, 4.0),
        hspace=0.04,
        wspace=0.04,
    )
    x_hist = figure.add_subplot(nested[0, 0])
    scatter = figure.add_subplot(nested[1, 0])
    y_hist = figure.add_subplot(nested[1, 1])
    x_limits, y_limits = _coordinate_limits(coordinates)

    x_hist.hist(
        coordinates[:, 0],
        bins=22,
        range=x_limits,
        density=True,
        color="#D9DDE2",
        edgecolor="none",
    )
    y_hist.hist(
        coordinates[:, 1],
        bins=22,
        range=y_limits,
        density=True,
        orientation="horizontal",
        color="#D9DDE2",
        edgecolor="none",
    )

    for group, (color, marker) in group_styles.items():
        draw = labels.eq(group)
        if not draw.any():
            continue
        points = coordinates[draw]
        group_name = group_names.get(group, f"group_{group}")
        scatter.scatter(
            points[:, 0],
            points[:, 1],
            s=7,
            alpha=0.48,
            color=color,
            marker=marker,
            linewidths=0.45,
            rasterized=True,
            label=group_name,
        )
        x_hist.hist(
            points[:, 0],
            bins=22,
            range=x_limits,
            density=True,
            histtype="step",
            color=color,
            linewidth=0.55,
            alpha=0.75,
        )
        y_hist.hist(
            points[:, 1],
            bins=22,
            range=y_limits,
            density=True,
            histtype="step",
            orientation="horizontal",
            color=color,
            linewidth=0.55,
            alpha=0.75,
        )

    scatter.set(xlim=x_limits, ylim=y_limits, xlabel="PC1", ylabel="PC2")
    scatter.spines[["top", "right"]].set_visible(False)
    scatter.tick_params(direction="out", length=2.5, pad=1.5)
    scatter.axhline(0.0, color="#D6D6D6", linewidth=0.45, zorder=0)
    scatter.axvline(0.0, color="#D6D6D6", linewidth=0.45, zorder=0)
    x_hist.set_title(title, fontsize=7, pad=3)
    x_hist.set_xlim(x_limits)
    y_hist.set_ylim(y_limits)
    for marginal in (x_hist, y_hist):
        marginal.set_xticks([])
        marginal.set_yticks([])
        for spine in marginal.spines.values():
            spine.set_visible(False)
    x_hist.text(
        -0.18,
        1.10,
        panel_letter,
        transform=x_hist.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def _group_styles(
    labels: Tensor,
    group_names: Mapping[int, str],
) -> dict[int, tuple[str, str]]:
    groups = sorted(int(group) for group in torch.unique(labels).tolist())
    regular = [
        group
        for group in groups
        if group_names.get(group, f"group_{group}") not in _SPECIAL_STYLES
    ]
    colors = {
        group: _GROUP_COLORS[index % len(_GROUP_COLORS)]
        for index, group in enumerate(regular)
    }
    return {
        group: _SPECIAL_STYLES.get(
            group_names.get(group, f"group_{group}"),
            (colors.get(group, _GROUP_COLORS[0]), "o"),
        )
        for group in groups
    }


def _safe_key(name: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name
    )


def _coordinate_limits(
    coordinates: Tensor,
) -> tuple[tuple[float, float], tuple[float, float]]:
    limits = []
    for dimension in range(2):
        values = coordinates[:, dimension]
        lower = float(torch.quantile(values, 0.01))
        upper = float(torch.quantile(values, 0.99))
        padding = max(0.05 * (upper - lower), 0.1)
        limits.append((lower - padding, upper + padding))
    return limits[0], limits[1]
