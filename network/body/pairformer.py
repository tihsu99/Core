import math

import torch
from torch import Tensor, nn


class PairBias(nn.Module):
    """Project pair features to one additive attention bias per head."""

    def __init__(self, input_dim: int, num_heads: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 2 * num_heads),
            nn.GELU(approximate="none"),
            nn.Linear(2 * num_heads, num_heads),
        )

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        bias = self.projection(pair)
        bias = bias * pair_mask.unsqueeze(-1).to(bias.dtype)
        return bias.permute(0, 3, 1, 2).contiguous()


class PairEmbedding(nn.Module):
    """Embed raw physics pair features into a persistent pair state."""

    def __init__(self, input_dim: int, pair_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 2 * pair_dim),
            nn.GELU(approximate="none"),
            nn.Linear(2 * pair_dim, pair_dim),
        )

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        pair = self.projection(pair)
        return pair * pair_mask.unsqueeze(-1).to(pair.dtype)


class PairToAttentionBias(nn.Module):
    """AlphaFold-style pair-state projection used by single attention."""

    def __init__(self, pair_dim: int, num_heads: int):
        super().__init__()
        self.normalization = nn.LayerNorm(pair_dim)
        self.projection = nn.Linear(pair_dim, num_heads, bias=False)

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        bias = self.projection(self.normalization(pair))
        bias = bias * pair_mask.unsqueeze(-1).to(bias.dtype)
        return bias.permute(0, 3, 1, 2).contiguous()


class ObjectToPair(nn.Module):
    """Project ordered object latents (h_i, h_j) into the pair state."""

    def __init__(self, object_dim: int, pair_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(2 * object_dim),
            nn.Linear(2 * object_dim, 2 * pair_dim),
            nn.GELU(approximate="none"),
            nn.Linear(2 * pair_dim, pair_dim),
        )

    def forward(self, objects: Tensor, pair_mask: Tensor) -> Tensor:
        num_objects = objects.shape[1]
        left = objects[:, :, None, :].expand(-1, -1, num_objects, -1)
        right = objects[:, None, :, :].expand(-1, num_objects, -1, -1)
        update = self.projection(torch.cat([left, right], dim=-1))
        return update * pair_mask.unsqueeze(-1).to(update.dtype)


class TriangleMultiplication(nn.Module):
    """Gated outgoing or incoming triangle multiplication on pair states."""

    def __init__(self, pair_dim: int, outgoing: bool):
        super().__init__()
        self.outgoing = outgoing
        self.normalization = nn.LayerNorm(pair_dim)
        self.left_projection = nn.Linear(pair_dim, pair_dim)
        self.right_projection = nn.Linear(pair_dim, pair_dim)
        self.left_gate = nn.Linear(pair_dim, pair_dim)
        self.right_gate = nn.Linear(pair_dim, pair_dim)
        self.output_gate = nn.Linear(pair_dim, pair_dim)
        self.output_projection = nn.Linear(pair_dim, pair_dim)

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        normalized = self.normalization(pair)
        mask = pair_mask.unsqueeze(-1).to(pair.dtype)
        left = self.left_projection(normalized) * torch.sigmoid(self.left_gate(normalized)) * mask
        right = self.right_projection(normalized) * torch.sigmoid(self.right_gate(normalized)) * mask

        if self.outgoing:
            update = torch.einsum("bikd,bjkd->bijd", left, right)
        else:
            update = torch.einsum("bkjd,bkid->bijd", left, right)

        update = update / math.sqrt(max(pair.shape[1], 1))
        update = self.output_projection(update)
        update = update * torch.sigmoid(self.output_gate(normalized))
        return update * mask


class TriangleAttention(nn.Module):
    """Attention over triangles sharing a starting node."""

    def __init__(self, pair_dim: int, num_heads: int):
        super().__init__()
        if pair_dim % num_heads != 0:
            raise ValueError(
                f"pair_dim ({pair_dim}) must be divisible by pair_num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.normalization = nn.LayerNorm(pair_dim)
        self.query = nn.Linear(pair_dim, pair_dim, bias=False)
        self.key = nn.Linear(pair_dim, pair_dim, bias=False)
        self.value = nn.Linear(pair_dim, pair_dim, bias=False)
        self.pair_bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.gate = nn.Linear(pair_dim, pair_dim)
        self.output = nn.Linear(pair_dim, pair_dim)

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _split_heads(self, value: Tensor) -> Tensor:
        batch_size, num_objects, _, _ = value.shape
        value = value.view(batch_size, num_objects, num_objects, self.num_heads, self.head_dim)
        return value.permute(0, 3, 1, 2, 4)

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        normalized = self.normalization(pair)
        query = self._split_heads(self.query(normalized)) * self.scale
        key = self._split_heads(self.key(normalized))
        value = self._split_heads(self.value(normalized))

        # For fixed i, z_ij attends over z_ik, with z_jk supplying the triangle bias.
        logits = torch.einsum("bhijd,bhikd->bhijk", query, key)
        triangle_bias = self.pair_bias(normalized).permute(0, 3, 1, 2)
        logits = logits + triangle_bias[:, :, None, :, :]

        key_mask = pair_mask[:, None, :, None, :]
        logits = logits.masked_fill(~key_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        update = torch.einsum("bhijk,bhikd->bhijd", weights, value)
        update = update.permute(0, 2, 3, 1, 4).contiguous().view_as(pair)
        update = update * torch.sigmoid(self.gate(normalized))
        update = self.output(update)
        return update * pair_mask.unsqueeze(-1).to(update.dtype)


class PairTransition(nn.Module):
    def __init__(self, pair_dim: int):
        super().__init__()
        self.normalization = nn.LayerNorm(pair_dim)
        self.transition = nn.Sequential(
            nn.Linear(pair_dim, 4 * pair_dim),
            nn.ReLU(),
            nn.Linear(4 * pair_dim, pair_dim),
        )
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        update = self.transition(self.normalization(pair))
        return update * pair_mask.unsqueeze(-1).to(update.dtype)


class PairUpdateBlock(nn.Module):
    """AlphaFold-inspired pair update with optional triangle attention."""

    def __init__(self, pair_dim: int, num_heads: int, use_triangle_attention: bool):
        super().__init__()
        self.outgoing_multiplication = TriangleMultiplication(pair_dim, outgoing=True)
        self.incoming_multiplication = TriangleMultiplication(pair_dim, outgoing=False)
        self.use_triangle_attention = use_triangle_attention

        if use_triangle_attention:
            self.starting_node_attention = TriangleAttention(pair_dim, num_heads)
            self.ending_node_attention = TriangleAttention(pair_dim, num_heads)

        self.transition = PairTransition(pair_dim)

    @staticmethod
    def _apply_mask(pair: Tensor, pair_mask: Tensor) -> Tensor:
        return pair * pair_mask.unsqueeze(-1).to(pair.dtype)

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        pair = pair + self.outgoing_multiplication(pair, pair_mask)
        pair = self._apply_mask(pair, pair_mask)
        pair = pair + self.incoming_multiplication(pair, pair_mask)
        pair = self._apply_mask(pair, pair_mask)

        if self.use_triangle_attention:
            pair = pair + self.starting_node_attention(pair, pair_mask)
            pair = self._apply_mask(pair, pair_mask)

            transposed_pair = pair.transpose(1, 2)
            transposed_mask = pair_mask.transpose(1, 2)
            transposed_pair = transposed_pair + self.ending_node_attention(
                transposed_pair,
                transposed_mask,
            )
            pair = self._apply_mask(transposed_pair.transpose(1, 2), pair_mask)

        pair = pair + self.transition(pair, pair_mask)
        return self._apply_mask(pair, pair_mask)
