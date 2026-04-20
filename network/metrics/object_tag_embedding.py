from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


@dataclass(frozen=True)
class ObjectTagEmbeddingConfig:
    enabled: bool = False
    group_name: str = "ObjectTag"
    label_feature: str = "Part_pdgId"
    max_points: int = 4000
    max_batches: int = 2
    log_every_n_epochs: int = 1
    alpha: float = 0.5
    marker_size: float = 8.0
    outlier_percentile_low: float = 0.5
    outlier_percentile_high: float = 99.5
    interested_labels: dict[int, str] | None = None
    others_label: str = "others"

    @classmethod
    def from_raw(cls, raw_cfg) -> "ObjectTagEmbeddingConfig":
        raw_cfg = raw_cfg or {}
        interested_labels = {
            int(key): str(value)
            for key, value in (raw_cfg.get("interested_labels", {}) or {}).items()
        }
        return cls(
            enabled=bool(raw_cfg.get("enable", False)),
            group_name=str(raw_cfg.get("group_name", "ObjectTag")),
            label_feature=str(raw_cfg.get("label_feature", "Part_pdgId")),
            max_points=int(raw_cfg.get("max_points", 4000)),
            max_batches=int(raw_cfg.get("max_batches", 2)),
            log_every_n_epochs=int(raw_cfg.get("log_every_n_epochs", 1)),
            alpha=float(raw_cfg.get("alpha", 0.5)),
            marker_size=float(raw_cfg.get("marker_size", 8.0)),
            outlier_percentile_low=float(raw_cfg.get("outlier_percentile_low", 0.5)),
            outlier_percentile_high=float(raw_cfg.get("outlier_percentile_high", 99.5)),
            interested_labels=interested_labels,
            others_label=str(raw_cfg.get("others_label", "others")),
        )


class ObjectTagEmbeddingLogger:
    def __init__(self, raw_cfg=None):
        self.config = ObjectTagEmbeddingConfig.from_raw(raw_cfg)
        self._configured = False
        self._disabled_reason = None
        self._object_tag_indices: tuple[int, ...] = ()
        self._object_tag_feature_names: tuple[str, ...] = ()
        self._label_feature_index: int | None = None
        self.reset()

    def reset(self) -> None:
        self._num_batches = 0
        self._embeddings: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def _configure_from_model(self, model) -> None:
        if self._configured or not self.config.enabled:
            return

        projected_feature_names = tuple(getattr(model.event_info, "projected_sequential_feature_names", ()))
        raw_feature_names = tuple(getattr(model.event_info, "raw_sequential_feature_names", ()))

        object_tag_indices = [
            index for index, name in enumerate(projected_feature_names)
            if name == self.config.group_name or name.startswith(f"{self.config.group_name}_")
        ]
        if not object_tag_indices:
            self._disabled_reason = f"group '{self.config.group_name}' not found"
            self._configured = True
            return

        if self.config.label_feature not in raw_feature_names:
            self._disabled_reason = f"label feature '{self.config.label_feature}' not found"
            self._configured = True
            return

        self._object_tag_indices = tuple(object_tag_indices)
        self._object_tag_feature_names = tuple(projected_feature_names[index] for index in object_tag_indices)
        self._label_feature_index = raw_feature_names.index(self.config.label_feature)
        self._configured = True

    def update(self, model, batch) -> None:
        if not self.config.enabled or self._num_batches >= self.config.max_batches:
            return

        self._configure_from_model(model)
        if self._disabled_reason is not None or self._label_feature_index is None:
            return

        with torch.no_grad():
            x = batch["x"].to(device=model.device)
            valid_mask = batch["x_mask"].to(device=model.device, dtype=torch.bool)
            mask = valid_mask.unsqueeze(-1)

            normalized_x = model.sequential_normalizer(x=x, mask=mask)
            projected_x = model.project_sequential_inputs(x=normalized_x, mask=mask)

            object_tag = projected_x[..., list(self._object_tag_indices)][valid_mask].detach().cpu()
            pdg_id = x[..., self._label_feature_index][valid_mask].detach().cpu()

            if object_tag.numel() == 0:
                return

            self._embeddings.append(object_tag)
            self._labels.append(torch.round(pdg_id).to(torch.int64))
            self._num_batches += 1

    def _prepare_points(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._embeddings:
            return None

        embeddings = torch.cat(self._embeddings, dim=0)
        labels = torch.cat(self._labels, dim=0)

        if embeddings.size(0) > self.config.max_points:
            indices = torch.randperm(embeddings.size(0))[: self.config.max_points]
            embeddings = embeddings[indices]
            labels = labels[indices]

        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        q = min(centered.size(0), centered.size(1), 2)
        if q <= 0:
            return None

        _, _, v = torch.pca_lowrank(centered, q=q)
        reduced = centered @ v[:, :q]
        if reduced.size(-1) == 1:
            reduced = torch.cat([reduced, torch.zeros_like(reduced)], dim=-1)

        return reduced.numpy(), labels.numpy()

    def _filter_plot_outliers(self, points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = self.config.outlier_percentile_low
        high = self.config.outlier_percentile_high
        if points.shape[0] == 0 or not (0.0 <= low < high <= 100.0):
            return points, labels

        lower = np.percentile(points, low, axis=0)
        upper = np.percentile(points, high, axis=0)
        keep_mask = np.all((points >= lower) & (points <= upper), axis=1)

        # Avoid dropping everything if the sampled cloud is too small or degenerate.
        if not np.any(keep_mask):
            return points, labels
        return points[keep_mask], labels[keep_mask]

    def _label_name(self, pdg_id: int) -> str:
        pdg_id = abs(int(pdg_id))
        if self.config.interested_labels is None:
            return self.config.others_label
        return self.config.interested_labels.get(pdg_id, self.config.others_label)

    def _make_figure(self, points: np.ndarray, labels: np.ndarray, epoch: int):
        display_labels = np.array([self._label_name(int(label)) for label in labels])

        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        cmap = plt.cm.get_cmap("tab20", len(np.unique(display_labels)))

        for index, label_name in enumerate(np.unique(display_labels)):
            mask = display_labels == label_name
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=self.config.marker_size,
                alpha=self.config.alpha,
                color=cmap(index),
                label=label_name,
                linewidths=0,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"ObjectTag PCA by PDG ID (epoch {epoch})")
        ax.grid(alpha=0.15)
        ax.legend(loc="best", fontsize=8, markerscale=2)
        fig.tight_layout()
        return fig

    def log(self, loggers, epoch: int, global_rank: int) -> None:
        if not self.config.enabled or global_rank != 0:
            self.reset()
            return

        if self.config.log_every_n_epochs <= 0 or ((epoch + 1) % self.config.log_every_n_epochs) != 0:
            self.reset()
            return

        prepared = self._prepare_points()
        if prepared is None:
            self.reset()
            return

        points, labels = prepared
        points, labels = self._filter_plot_outliers(points, labels)
        if points.shape[0] == 0:
            self.reset()
            return

        fig = self._make_figure(points, labels, epoch)
        for logger in loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is None or not hasattr(experiment, "log"):
                continue
            experiment.log({"diagnostics/object_tag_embedding": wandb.Image(fig)})
            break
        plt.close(fig)
        self.reset()
