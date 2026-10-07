"""Truth-bond contrastive supervision for EveNet pair states."""

from fnmatch import fnmatchcase
from typing import Mapping

import torch
from torch import Tensor


LEVELS = {
    "heavy_ancestor": 0,
    "decay_system": 1,
    "sibling": 2,
}
BUCKETS = {
    "sibling": 0,
    "decay_system": 1,
    "heavy_ancestor": 2,
    "unrelated": 3,
}


class PairRelationLabeler:
    """Map EveNet assignment truth onto relations and mother categories."""

    def __init__(
        self,
        event_info,
        process_info: Mapping,
        truth_mother_categories: Mapping[str, list[str]],
        heavy_ancestor_groups: Mapping[str, list[list[str]]] | None = None,
    ) -> None:
        """Build the truth lookup tables used to label reconstructed pairs.

        Args:
            event_info: EveNet ``EventInfo`` object. It must provide:
                ``process_names`` as ``list[str]``;
                ``assignment_names[process]`` as the ordered truth roots; and
                ``product_particles[process][root].names`` as the ordered
                truth daughters stored in the assignment tensor.
            process_info: Mapping from each process name to its nested truth
                decay tree. The tree is read from
                ``process_info[process]["diagram"]``. Leaves are truth-particle
                names and ``SYMMETRY`` entries are ignored. Example::

                    {
                        "TT1L": {
                            "diagram": {
                                "t1": {"b": None, "W": {"l": None}}
                            }
                        }
                    }

            truth_mother_categories: Mapping from a canonical truth-mother
                category to every topology alias representing that category.
                Category IDs follow this mapping's insertion order. Every direct
                parent and every top-level root used by ``event_info`` must appear
                exactly once. Example::

                    {
                        "top": ["t", "t1", "t2"],
                        "W": ["W", "W1", "W2", "W+", "W-"],
                        "Z": ["Z", "Z1", "Z2"],
                    }

            heavy_ancestor_groups: Optional mapping from process-name glob
                patterns to groups of top-level roots that share one heavy
                ancestor. Roots not listed in a group remain separate. Example::

                    {"HWW_*": [["W+", "W-"]]}
        """
        # Convert the readable YAML form {category: [aliases]} into the lookup
        # used while traversing each truth path: {alias: integer_category_id}.
        mother_categories = {
            alias: category_id
            for category_id, aliases in enumerate(truth_mother_categories.values())
            for alias in aliases
        }
        if not mother_categories:
            raise ValueError("PairContrastive.truth_mother_categories must not be empty")
        if len(mother_categories) != sum(
            len(aliases) for aliases in truth_mother_categories.values()
        ):
            raise ValueError("Each truth mother alias must belong to one category")

        self.layouts: dict[int, list[tuple[int, int, int, int, int, int, int]]] = {}
        heavy_ancestor_groups = heavy_ancestor_groups or {}
        global_row = 0

        for process_id, process in enumerate(event_info.process_names):
            roots = list(event_info.assignment_names[process])
            process_cfg = process_info.get(process)
            if process_cfg is None:
                raise ValueError(f"Missing process_info topology for {process}")
            diagram = process_cfg.get("diagram", process_cfg)
            root_to_heavy = self._build_heavy_groups(
                process,
                roots,
                heavy_ancestor_groups,
            )
            direct_ids: dict[tuple[str, ...], int] = {}
            layout = []

            for local_id, root in enumerate(roots):
                if root not in diagram:
                    raise ValueError(f"process_info.{process} is missing root {root}")
                paths_by_leaf: dict[str, list[tuple[str, ...]]] = {}
                for path in self._leaf_paths(diagram[root], (root,)):
                    paths_by_leaf.setdefault(path[-1], []).append(path)

                product_names = event_info.product_particles[process][root].names
                for daughter_id, leaf in enumerate(product_names):
                    paths = paths_by_leaf.get(leaf, [])
                    if len(paths) != 1:
                        raise ValueError(
                            f"Expected one canonical path for {process}/{root}/{leaf}, "
                            f"found {len(paths)}"
                        )
                    path = paths[0]
                    parent_path = path[:-1]
                    direct_name = parent_path[-1]
                    missing_categories = [
                        name for name in (direct_name, root)
                        if name not in mother_categories
                    ]
                    if missing_categories:
                        raise ValueError(
                            "PairContrastive.truth_mother_categories does not classify "
                            f"{process}/{root}: {missing_categories}"
                        )
                    direct_id = direct_ids.setdefault(parent_path, len(direct_ids))
                    layout.append(
                        (
                            global_row,
                            daughter_id,
                            direct_id,
                            local_id,
                            root_to_heavy[root],
                            mother_categories[direct_name],
                            mother_categories[root],
                        )
                    )
                global_row += 1
            self.layouts[process_id] = layout
        self.num_assignment_rows = global_row
        self.num_assignment_daughters = max(
            (daughter + 1 for layout in self.layouts.values() for _, daughter, *_ in layout),
            default=0,
        )

    @staticmethod
    def _leaf_paths(node, prefix: tuple[str, ...]):
        if not isinstance(node, Mapping):
            yield prefix
            return
        for name, children in node.items():
            if name == "SYMMETRY":
                continue
            yield from PairRelationLabeler._leaf_paths(children, prefix + (name,))

    @staticmethod
    def _build_heavy_groups(
        process: str,
        roots: list[str],
        configured_groups: Mapping[str, list[list[str]]],
    ) -> dict[str, int]:
        groups = [
            group
            for pattern, pattern_groups in configured_groups.items()
            if fnmatchcase(process, pattern)
            for group in pattern_groups
        ]
        result = {}
        for group_id, group in enumerate(groups):
            for root in group:
                if root not in roots:
                    raise ValueError(
                        f"Unknown heavy-ancestor root {root} for process {process}"
                    )
                if root in result:
                    raise ValueError(
                        f"Root {process}/{root} belongs to multiple heavy ancestors"
                    )
                result[root] = group_id
        next_id = len(groups)
        for root in roots:
            if root not in result:
                result[root] = next_id
                next_id += 1
        return result

    def build_labels(
        self,
        assignments: Tensor,
        assignment_index_mask: Tensor,
        process_ids: Tensor,
        pair_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Return valid upper-triangle pair labels in ``[A, B, S]`` order."""
        if assignments.ndim != 3 or assignment_index_mask.shape != assignments.shape:
            raise ValueError("Assignment tensors must share shape [B, R, K]")
        if (
            assignments.shape[1] < self.num_assignment_rows
            or assignments.shape[2] < self.num_assignment_daughters
        ):
            raise ValueError("Assignment tensor is smaller than the configured truth topology")
        batch_size, num_objects, _ = pair_mask.shape
        if assignments.shape[0] != batch_size or process_ids.numel() != batch_size:
            raise ValueError("Assignment targets and pair state use different batches")

        device = assignments.device
        instance_metadata = [
            torch.full((batch_size, num_objects), -1, dtype=torch.long, device=device)
            for _ in range(3)
        ]
        category_metadata = [
            torch.full((batch_size, num_objects), -1, dtype=torch.long, device=device)
            for _ in range(2)
        ]
        match_count = torch.zeros(
            (batch_size, num_objects), dtype=torch.long, device=device
        )

        for process_id, layout in self.layouts.items():
            event_indices = torch.nonzero(
                process_ids.reshape(-1).long().eq(process_id), as_tuple=False
            ).flatten()
            if event_indices.numel() == 0:
                continue
            for (
                row,
                daughter,
                direct_id,
                local_id,
                heavy_id,
                direct_category,
                local_category,
            ) in layout:
                object_indices = assignments[event_indices, row, daughter].long()
                valid = (
                    assignment_index_mask[event_indices, row, daughter].bool()
                    & object_indices.ge(0)
                    & object_indices.lt(num_objects)
                )
                event = event_indices[valid]
                obj = object_indices[valid]
                if event.numel() == 0:
                    continue
                match_count.index_put_(
                    (event, obj),
                    torch.ones_like(event),
                    accumulate=True,
                )
                for target, value in zip(
                    instance_metadata,
                    (direct_id, local_id, heavy_id),
                ):
                    target[event, obj] = value
                for target, value in zip(
                    category_metadata,
                    (direct_category, local_category),
                ):
                    target[event, obj] = value

        object_valid = match_count.eq(1)
        direct, local, heavy = instance_metadata
        direct_category, local_category = category_metadata
        valid_pairs = (
            pair_mask.bool()
            & object_valid.unsqueeze(2)
            & object_valid.unsqueeze(1)
            & torch.ones(
                (num_objects, num_objects), dtype=torch.bool, device=device
            ).triu(diagonal=1).unsqueeze(0)
        )
        same_direct = direct.unsqueeze(2).eq(direct.unsqueeze(1)) & valid_pairs
        same_local = local.unsqueeze(2).eq(local.unsqueeze(1)) & valid_pairs
        same_heavy = heavy.unsqueeze(2).eq(heavy.unsqueeze(1)) & valid_pairs
        inconsistent = (same_direct & ~same_local) | (same_local & ~same_heavy)
        valid_pairs = valid_pairs & ~inconsistent
        relations = torch.stack(
            (same_heavy, same_local, same_direct), dim=-1
        ) & valid_pairs.unsqueeze(-1)
        categories = torch.full(
            (*valid_pairs.shape, len(LEVELS)), -1, dtype=torch.long, device=device
        )
        categories[..., LEVELS["decay_system"]] = torch.where(
            same_local,
            local_category.unsqueeze(2).expand(-1, -1, num_objects),
            -1,
        )
        categories[..., LEVELS["sibling"]] = torch.where(
            same_direct,
            direct_category.unsqueeze(2).expand(-1, -1, num_objects),
            -1,
        )

        bucket = torch.full(
            valid_pairs.shape, -1, dtype=torch.long, device=device
        )
        bucket[valid_pairs & ~relations[..., 0]] = BUCKETS["unrelated"]
        bucket[relations[..., 0] & ~relations[..., 1]] = BUCKETS["heavy_ancestor"]
        bucket[relations[..., 1] & ~relations[..., 2]] = BUCKETS["decay_system"]
        bucket[relations[..., 2]] = BUCKETS["sibling"]
        return {
            "relations": relations,
            "categories": categories,
            "bucket": bucket,
            "valid": valid_pairs,
            "num_inconsistent_pairs": inconsistent.sum(),
        }

    @staticmethod
    def select_valid_pairs(labels: dict[str, Tensor]) -> dict[str, Tensor]:
        """Flatten every valid local pair and its truth metadata."""
        batch, i, j = torch.nonzero(labels["valid"], as_tuple=True)
        return {
            "batch": batch,
            "i": i,
            "j": j,
            "relations": labels["relations"][batch, i, j],
            "categories": labels["categories"][batch, i, j],
            "bucket": labels["bucket"][batch, i, j],
            "event": batch,
            "pair_id": torch.arange(batch.numel(), device=batch.device),
            "num_inconsistent_pairs": labels["num_inconsistent_pairs"],
        }


def build_contrastive_masks(
    local: dict[str, Tensor], index: Tensor, level: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return [anchors, pool] masks shared by the loss and its diagnostics.

    Outputs are same-event, positive, endpoint-negative, and random-negative
    candidate masks. Random candidates have not yet been sampled.
    """
    same_event = local["event"][index, None].eq(local["event"][None, :])
    same_pair = local["pair_id"][index, None].eq(local["pair_id"][None, :])
    same_category = local["categories"][index, level, None].eq(
        local["categories"][None, :, level]
    )
    same_bond = same_category & local["relations"][:, level].unsqueeze(0)
    positive = same_bond & ~same_pair
    shares_endpoint = (
        local["i"][index, None].eq(local["i"][None, :])
        | local["i"][index, None].eq(local["j"][None, :])
        | local["j"][index, None].eq(local["i"][None, :])
        | local["j"][index, None].eq(local["j"][None, :])
    )
    truth_negative = ~same_bond & ~same_pair
    endpoint_negative = same_event & shares_endpoint & truth_negative
    random_candidates = same_event & ~shares_endpoint & truth_negative
    return same_event, positive, endpoint_negative, random_candidates


class HierarchicalPairContrastiveLoss:
    """Parameter-free local truth-bond contrastive objective."""

    def __init__(self, config) -> None:
        self.temperature = float(config.get("temperature", 0.1))
        if self.temperature <= 0:
            raise ValueError("PairContrastive.temperature must be positive")
        self.candidate_balance = config.get("candidate_balance", "none")
        if self.candidate_balance not in ("none", "positive_negative"):
            raise ValueError("PairContrastive.candidate_balance must be none or positive_negative")
        self.level_weights = config.get("level_weights", {})
        self.enabled_levels = tuple(
            config.get("enabled_levels", ("decay_system", "sibling"))
        )
        unsupported = set(self.enabled_levels) - {"decay_system", "sibling"}
        if unsupported:
            raise ValueError(
                "Truth-bond contrastive levels must be decay_system and/or sibling; "
                f"got {sorted(unsupported)}"
            )
        if not self.enabled_levels:
            raise ValueError("PairContrastive.enabled_levels must not be empty")
        self.max_random_negatives_per_anchor = int(
            config.get("max_random_negatives_per_anchor", 64)
        )
        self.anchor_chunk_size = int(config.get("anchor_chunk_size", 256))
        if self.max_random_negatives_per_anchor < 0 or self.anchor_chunk_size < 1:
            raise ValueError(
                "PairContrastive random-negative cap must be non-negative and "
                "anchor_chunk_size must be positive"
            )

    def __call__(
        self,
        z: Tensor,
        selected: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        losses = {}
        counts = {
            "num_same_event_positives": z.new_zeros(()),
            "num_cross_event_positives": z.new_zeros(()),
            "num_endpoint_negatives": z.new_zeros(()),
            "num_random_negatives": z.new_zeros(()),
        }
        total = z.sum() * 0.0
        for name in self.enabled_levels:
            level = LEVELS[name]
            counts[f"num_candidate_anchors_{name}"] = (
                selected["relations"][:, level]
                & selected["categories"][:, level].ge(0)
            ).sum()
            level_loss, level_count, level_counts = self._level_loss(
                z, selected, level
            )
            losses[name] = level_loss
            counts[f"num_valid_anchors_{name}"] = level_count
            for key, value in level_counts.items():
                counts[key] = counts[key] + value
                counts[f"{key}_{name}"] = value
            total = total + float(self.level_weights.get(name, 1.0)) * level_loss

        metrics = {
            "loss": total.detach(),
            **{f"loss_{name}": value.detach() for name, value in losses.items()},
            "num_local_pairs": z.new_tensor(float(z.shape[0])),
            "num_inconsistent_pairs": selected["num_inconsistent_pairs"].to(z.dtype),
            **counts,
        }
        for name, bucket_id in BUCKETS.items():
            metrics[f"fraction_{name}"] = selected["bucket"].eq(bucket_id).float().mean() \
                if z.shape[0] else z.new_zeros(())
        return total, metrics

    def _level_loss(
        self,
        z: Tensor,
        local: dict[str, Tensor],
        level: int,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        anchors = torch.nonzero(
            local["relations"][:, level] & local["categories"][:, level].ge(0),
            as_tuple=False,
        ).flatten()
        loss_sum = z.sum() * 0.0
        valid_count = 0
        counts = {
            "num_same_event_positives": z.new_zeros(()),
            "num_cross_event_positives": z.new_zeros(()),
            "num_endpoint_negatives": z.new_zeros(()),
            "num_random_negatives": z.new_zeros(()),
        }

        for start in range(0, anchors.numel(), self.anchor_chunk_size):
            index = anchors[start:start + self.anchor_chunk_size]
            logits = z[index] @ z.transpose(0, 1) / self.temperature
            same_event, positive, endpoint_negative, random_candidates = (
                build_contrastive_masks(local, index, level)
            )
            random_negative = self._sample_random_negatives(random_candidates)
            negative = endpoint_negative | random_negative
            valid = positive.any(dim=1) & negative.any(dim=1)
            if not valid.any():
                continue
            # Filter before logsumexp: empty groups must not produce NaN gradients.
            per_anchor = self._anchor_losses(
                logits[valid], positive[valid], negative[valid],
            )
            loss_sum = loss_sum + per_anchor.sum()
            valid_count += int(valid.sum())
            counts["num_same_event_positives"] += (
                positive[valid] & same_event[valid]
            ).sum()
            counts["num_cross_event_positives"] += (
                positive[valid] & ~same_event[valid]
            ).sum()
            counts["num_endpoint_negatives"] += endpoint_negative[valid].sum()
            counts["num_random_negatives"] += random_negative[valid].sum()

        count = z.new_tensor(float(valid_count))
        return loss_sum / count.clamp_min(1.0), count, counts

    def _anchor_losses(self, logits: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Losses for anchors with nonempty, disjoint positive/negative groups.

        positive_negative adapts BCL Eq. (6)'s group-count averaging to two
        anchor-relative groups, not semantic classes; it is not full BCL.
        Reference: https://arxiv.org/html/2207.09052v3#S3.SS3
        """
        if self.candidate_balance == "positive_negative":
            # Keep log reductions stable under mixed precision without lowering float64.
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            positive_count = positive.sum(dim=1).to(logits.dtype)
            negative_count = negative.sum(dim=1).to(logits.dtype)
            log_positive_mean = torch.logsumexp(
                logits.masked_fill(~positive, -torch.inf), dim=1,
            ) - positive_count.log()
            log_negative_mean = torch.logsumexp(
                logits.masked_fill(~negative, -torch.inf), dim=1,
            ) - negative_count.log()
            log_denominator = torch.logaddexp(log_positive_mean, log_negative_mean)
        else:
            log_denominator = torch.logsumexp(
                logits.masked_fill(~(positive | negative), -torch.inf), dim=1,
            )
        log_probability = logits - log_denominator.unsqueeze(1)
        return -(
            log_probability.masked_fill(~positive, 0.0).sum(dim=1)
            / positive.sum(dim=1)
        )

    def _sample_random_negatives(
        self, candidates: Tensor, generator: torch.Generator | None = None,
    ) -> Tensor:
        """Keep at most the configured number of random negatives per anchor."""
        cap = self.max_random_negatives_per_anchor
        if cap == 0:
            return torch.zeros_like(candidates)
        if candidates.shape[1] <= cap:
            return candidates

        scores = torch.rand(
            candidates.shape, device=candidates.device, generator=generator
        )
        scores.masked_fill_(~candidates, 2.0)
        selected = scores.topk(cap, dim=1, largest=False).indices
        sampled = torch.zeros_like(candidates)
        sampled.scatter_(1, selected, True)
        return sampled & candidates
