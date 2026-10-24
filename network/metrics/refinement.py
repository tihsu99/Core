"""Matched, unweighted validation metrics for every recycling depth."""

import torch

from evenet.network.loss.segmentation import hungarian_matching


class RefinementMetrics:
    def __init__(self, max_recycles, num_queries, device, process_names=(), mask_threshold=0.5):
        if not 0 < mask_threshold < 1:
            raise ValueError("Refinement mask_threshold must be between zero and one")
        self.max_recycles = max_recycles
        self.num_queries = num_queries
        self.process_names = tuple(process_names)
        self.mask_threshold = mask_threshold
        self.groups = (
            [""] + [f"process/{name}/" for name in self.process_names]
            + [f"segments/{count}/" for count in range(num_queries + 1)]
        )
        self.totals = torch.zeros(
            len(self.groups), max_recycles + 1, 11, dtype=torch.float64, device=device
        )

    def reset(self):
        self.totals.zero_()

    @torch.no_grad()
    def update(self, steps, target_classes, target_masks, object_mask, process_ids=None):
        expected = {f"iteration_{depth}" for depth in range(self.max_recycles + 1)}
        if set(steps) != expected:
            raise ValueError("Refinement monitoring requires predictions at every configured depth")
        valid = object_mask.bool()
        if valid.ndim == 3:
            valid = valid.squeeze(-1)
        target_classes = target_classes.float()
        target_masks = target_masks.float().masked_fill(~valid[:, None, :], 0)
        target_labels = target_classes.argmax(-1)
        counts = (target_labels != 0).sum(-1)
        if target_classes.shape[1] != self.num_queries:
            raise ValueError("Refinement target slots must match Segmentation.num_queries")
        usable = valid.any(-1)
        selections = [usable]
        if self.process_names:
            if process_ids is None:
                raise ValueError("Refinement monitoring requires subprocess_id for process metrics")
            process_ids = process_ids.reshape(-1)
            selections += [usable & (process_ids == index) for index in range(len(self.process_names))]
        selections += [usable & (counts == count) for count in range(self.num_queries + 1)]
        selections = torch.stack(selections).to(self.totals.dtype)
        baseline = previous = None
        for depth in range(self.max_recycles + 1):
            prediction = steps[f"iteration_{depth}"]
            logits = prediction["pred_logits"]
            mask_logits = prediction["pred_masks"].float().masked_fill(~valid[:, None, :], -1e4)
            pred_indices, target_indices = hungarian_matching(
                predict_cls=logits.float(), predict_mask=mask_logits,
                target_cls=target_classes, target_mask=target_masks,
            )
            batch = torch.arange(valid.shape[0], device=valid.device)[:, None]
            probability = mask_logits[batch, pred_indices].sigmoid()
            truth = target_masks[batch, target_indices].bool()
            truth_labels = target_labels[batch, target_indices]
            positive = truth_labels != 0
            class_correct = logits[batch, pred_indices].argmax(-1) == truth_labels
            predicted = probability > self.mask_threshold
            exact_mask = (predicted == truth).all(-1)
            intersection = (probability * truth).sum(-1)
            total = probability.sum(-1) + truth.sum(-1)
            dice = (2 * intersection + 1e-8) / (total + 1e-8)
            iou = (intersection + 1e-8) / (total - intersection + 1e-8)
            exact_segment = class_correct & exact_mask & positive
            # Null target slots must stay null, including events with no target instances.
            exact_event = (class_correct & (exact_mask | ~positive)).all(-1)
            if baseline is None:
                baseline = previous = exact_event
            values = torch.stack([
                torch.ones_like(counts), counts,
                (dice * positive).sum(-1), (iou * positive).sum(-1),
                (class_correct & positive).sum(-1), exact_segment.sum(-1), exact_event,
                exact_event & ~baseline, ~exact_event & baseline,
                exact_event & ~previous, ~exact_event & previous,
            ], dim=-1).to(self.totals.dtype)
            self.totals[:, depth] += selections @ values
            previous = exact_event

    def compute(self):
        totals = self.totals.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        result = {}
        segment_metrics = ("dice", "iou", "class_accuracy", "exact_segment_accuracy")
        event_metrics = (
            "exact_event_accuracy", "corrected_vs_0", "damaged_vs_0",
            "corrected_vs_previous", "damaged_vs_previous",
        )
        for group, values in zip(self.groups, totals):
            for depth, row in enumerate(values):
                prefix = f"refinement/val/{group}iteration_{depth}/"
                events, segments = row[:2]
                result[prefix + "events"] = events
                result[prefix + "segments"] = segments
                if segments > 0:
                    for index, name in enumerate(segment_metrics, start=2):
                        result[prefix + name] = row[index] / segments
                if events > 0:
                    for index, name in enumerate(event_metrics, start=6):
                        result[prefix + name] = row[index] / events
                    result[prefix + "net_corrected_vs_0"] = (row[7] - row[8]) / events
        return result
