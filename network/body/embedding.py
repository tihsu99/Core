from typing import Any, Optional

from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module

from evenet.network.layers.linear_block import create_linear_block
from evenet.network.layers.transformer import TransformerBlockModule
from evenet.network.layers.utils import RandomDrop
import torch
from evenet.network.body.adapter import Adapter
from evenet.network.body.pairformer import (
    PairBias,
    PairEmbedding,
    PairToAttentionBias,
    PairUpdateBlock,
)

class EmbeddingStack(nn.Module):
    def __init__(self, linear_block_type: str,
                 input_dim: int,
                 hidden_dim_scale: float,
                 initial_embedding_dim: int,
                 final_embedding_dim: int,
                 normalization_type: str,
                 activation_type: str,
                 dropout: float,
                 skip_connection: bool,
                 num_embedding_layers: int):
        super(EmbeddingStack, self).__init__()
        self.input_dim = input_dim
        self.embedding_layers = nn.ModuleList(
            self.create_embedding_layers(
                linear_block_type=linear_block_type,
                input_dim=input_dim,
                hidden_dim_scale=hidden_dim_scale,
                initial_embedding_dim=initial_embedding_dim,
                final_embedding_dim=final_embedding_dim,
                normalization_type=normalization_type,
                activation_type=activation_type,
                dropout=dropout,
                skip_connection=skip_connection,
                num_embedding_layers=num_embedding_layers
            )
        )

    @staticmethod
    def create_embedding_layers(linear_block_type: str,
                                input_dim: int,
                                hidden_dim_scale: float,
                                initial_embedding_dim: int,
                                final_embedding_dim: int,
                                normalization_type: str,
                                activation_type: str,
                                dropout: float,
                                skip_connection: bool,
                                num_embedding_layers: int) -> list[Module]:
        embedding_layers = [create_linear_block(linear_block_type=linear_block_type,
                                                input_dim=input_dim,
                                                hidden_dim_scale=hidden_dim_scale,
                                                output_dim=initial_embedding_dim,
                                                normalization_type=normalization_type,
                                                activation_type=activation_type,
                                                dropout=dropout,
                                                skip_connection=skip_connection)]

        current_embedding_dim = initial_embedding_dim
        for i in range(num_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim > final_embedding_dim:
                break
            embedding_layers.append(create_linear_block(linear_block_type=linear_block_type,
                                                        input_dim=current_embedding_dim,
                                                        hidden_dim_scale=hidden_dim_scale,
                                                        output_dim=next_embedding_dim,
                                                        normalization_type=normalization_type,
                                                        activation_type=activation_type,
                                                        dropout=dropout,
                                                        skip_connection=skip_connection))
            current_embedding_dim = next_embedding_dim

        embedding_layers.append(create_linear_block(linear_block_type=linear_block_type,
                                                    input_dim=current_embedding_dim,
                                                    hidden_dim_scale=hidden_dim_scale,
                                                    output_dim=final_embedding_dim,
                                                    normalization_type=normalization_type,
                                                    activation_type=activation_type,
                                                    dropout=dropout,
                                                    skip_connection=skip_connection))

        return embedding_layers

    def forward(self, vectors: Tensor, mask: Tensor) -> Tensor:
        """
        :param vectors: shape: (batch_size, num_object, input_dim)
        :param mask: (batch_size, num_object, 1)
        :return:
            - output: shape (batch_size, num_object, final_embedding_dim)

        """
        embeddings = vectors
        for layer in self.embedding_layers:
            embeddings = layer(
                x=embeddings,
                sequence_mask=mask
            )
        return embeddings


class GlobalVectorEmbedding(nn.Module):
    def __init__(self,
                 linear_block_type: str,
                 input_dim: int,
                 hidden_dim_scale: float,
                 initial_embedding_dim: int,
                 final_embedding_dim: int,
                 normalization_type: str,
                 activation_type: str,
                 dropout: float,
                 skip_connection: bool,
                 num_embedding_layers: int
                 ):
        super(GlobalVectorEmbedding, self).__init__()
        self.embedding_stack = EmbeddingStack(linear_block_type=linear_block_type,
                                              input_dim=input_dim,
                                              hidden_dim_scale=hidden_dim_scale,
                                              initial_embedding_dim=initial_embedding_dim,
                                              final_embedding_dim=final_embedding_dim,
                                              normalization_type=normalization_type,
                                              activation_type=activation_type,
                                              dropout=dropout,
                                              skip_connection=skip_connection,
                                              num_embedding_layers=num_embedding_layers
                                              )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # --------------------------------
        # Embed vectors into latent space.
        # output: (max_vectors, batch_size, final_embedding_dim)
        # --------------------------------
        embeddings = x
        encoded = self.embedding_stack(
            vectors=embeddings,
            mask=mask
        )

        # ----------------------------
        # Local Embedding (For Point-Edge Point Cloud)
        # output: [T, B, D]
        # ----------------------------

        return encoded


class FourierEmbedding(nn.Module):
    def __init__(self, projection_dim, num_embed=64):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_embed = num_embed
        self.half_dim = num_embed // 2

        # Calculate frequencies
        emb = torch.log(torch.tensor(10000.0)) / (self.half_dim - 1)
        self.freq = torch.exp(-emb * torch.arange(self.half_dim, dtype=torch.float32))

        self.dense1 = nn.Linear(num_embed, 2 * projection_dim, bias=False)
        self.dense2 = nn.Linear(2 * projection_dim, projection_dim, bias=False)

    def forward(self, x):
        """

        :param x: time, shape (batch_size, 1)
        :return:
            embedding: shape (batch_size, projection_dim)
        """
        # To Ensure x is 2D: (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)

        angle = x * self.freq.to(x.device) * 1000.0
        embedding = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1) * x

        embedding = self.dense1(embedding)
        embedding = F.silu(embedding)  # SiLU is equivalent to Swish
        embedding = self.dense2(embedding)
        embedding = F.silu(embedding)

        return embedding


class LocalEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, projection_dim, K, num_local=2):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.K = K
        self.num_local = num_local
        self.local_embed_layer = nn.ModuleList([(LocalEmbeddingBlock(self.input_dim, self.projection_dim,
                                                                     self.K) if i == 0 else LocalEmbeddingBlock(
            self.projection_dim, self.projection_dim, self.K)) for i in range(self.num_local)])

    def forward(self, x: Tensor, points: Tensor, mask: Tensor) -> Tensor:
        """

        :param x: shape: (batch_size, num_points, input_dim)
        :param points: shape: (batch_size, num_points, num_features)
        :param mask: shape: (batch_size, num_points, 1)
        :return:
        """
        mask = mask.to(torch.int)
        coord_shift = 999.0 * (mask == 0).float()
        local_features = x
        for idx, local_embed in enumerate(self.local_embed_layer):
            local_features = local_embed(
                points=coord_shift + points,
                features=local_features
            )  # [B, T, D]
            points = local_features  # tihsu TODO: add mask ?

        return local_features * mask.float()


class LocalEmbeddingBlock(nn.Module):
    def __init__(self, input_dim, projection_dim, K):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, 2 * self.projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2 * self.projection_dim, self.projection_dim),
            nn.GELU(approximate='none')
        )

    def pairwise_distance(self, points):
        r = torch.sum(points * points, dim=2, keepdim=True)  # [B, T, D]
        m = torch.bmm(points, points.transpose(1, 2))  # [B, T, D] x [B, D, T] -> [B, T, T]
        D = r - 2 * m + r.transpose(1, 2) + 1e-5
        return D

    def forward(self, points, features):
        distances = self.pairwise_distance(points)  # uses custom pairwise function, not torch.cdist
        _, indices = torch.topk(-distances, k=self.K + 1, dim=-1)
        indices = indices[:, :, 1:]  # Exclude self
        # indices Shape: (N, P, 10)

        batch_size, num_points, _ = features.shape
        batch_indices = torch.arange(batch_size, device=features.device).view(-1, 1, 1)
        batch_indices = batch_indices.repeat(1, num_points, self.K)
        indices = torch.stack([batch_indices, indices], dim=-1)
        # concat indices torch.Size([N, P, K, 2])

        # Gather neighbor features
        neighbors = features[
            indices[:, :, :, 0], indices[:, :, :, 1]]  # Shape: (N, P, K, C) | neighbors: torch.Size([64, 150, 10, 13])
        knn_fts_center = features.unsqueeze(2).expand_as(
            neighbors)  # Shape: (N, P, K, C) | knn fts center: torch.Size([64, 150, 10, 13])
        local_features = torch.cat([neighbors - knn_fts_center, knn_fts_center],
                                   dim=-1)  # local_features: torch.Size([N, P, K, 26]) local_features.shape[-1] Shape : 2*C

        local_features = self.mlp(local_features)
        local_features = torch.mean(local_features, dim=2)

        return local_features


class PETBody(nn.Module):
    def __init__(
            self, num_feat, num_keep, feature_drop, projection_dim, local, K, num_local,
            num_layers, num_heads, drop_probability, talking_head, layer_scale,
            layer_scale_init, dropout, mode, use_adapter: bool = False,
            attention_bias_type: Optional[str] = None,
            pair_input_dim: int = 0,
            pair_dim: Optional[int] = None,
            pair_num_heads: Optional[int] = None,
            use_triangle_attention: bool = False,
    ):
        super().__init__()
        if K < 0:
            raise ValueError(f"K must be non-negative, got {K}.")
        if attention_bias_type not in (None, "SimpleAddition", "IterativeUpdate"):
            raise ValueError(
                "attention_bias_type must be None, 'SimpleAddition', or 'IterativeUpdate'."
            )
        if attention_bias_type is not None and pair_input_dim <= 0:
            raise ValueError(
                f"{attention_bias_type} requires a positive pair_input_dim."
            )
        if use_triangle_attention and attention_bias_type != "IterativeUpdate":
            raise ValueError(
                "use_triangle_attention is only valid with attention_bias_type='IterativeUpdate'."
            )

        self.num_keep = num_keep
        self.feature_drop = feature_drop
        self.projection_dim = projection_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_local = num_local
        self.drop_probability = drop_probability
        self.layer_scale = layer_scale
        self.layer_scale_init = layer_scale_init
        self.dropout = dropout
        self.attention_bias_type = attention_bias_type
        self.pair_input_dim = pair_input_dim
        self.use_triangle_attention = use_triangle_attention
        self.mode = mode

        self.random_drop = RandomDrop(feature_drop if 'all' in self.mode else 0.0, num_keep)
        self.feature_embedding = nn.Sequential(
            nn.Linear(num_feat, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2 * projection_dim, projection_dim),
            nn.GELU(approximate='none')
        )

        self.time_embedding = FourierEmbedding(projection_dim)
        self.time_embed_linear = nn.Linear(projection_dim, 2 * projection_dim, bias=False)

        # K == 0 explicitly disables the kNN branch, even when local embedding is enabled.
        self.use_local_embedding = bool(local and K > 0)
        if self.use_local_embedding:
            self.local_embedding = LocalEmbeddingLayer(num_feat, projection_dim, K, num_local)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlockModule(
                projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init,
                drop_probability
            )
            for _ in range(num_layers)
        ])

        if self.attention_bias_type == "SimpleAddition":
            self.simple_pair_bias = PairBias(pair_input_dim, num_heads)
        elif self.attention_bias_type == "IterativeUpdate":
            pair_dim = projection_dim if pair_dim is None else pair_dim
            pair_num_heads = num_heads if pair_num_heads is None else pair_num_heads
            if pair_dim <= 0:
                raise ValueError(f"pair_dim must be positive, got {pair_dim}.")

            self.pair_embedding = PairEmbedding(pair_input_dim, pair_dim)
            self.pair_update_blocks = nn.ModuleList([
                PairUpdateBlock(
                    pair_dim=pair_dim,
                    num_heads=pair_num_heads,
                    use_triangle_attention=use_triangle_attention,
                )
                for _ in range(num_layers)
            ])
            self.iterative_pair_bias = PairToAttentionBias(pair_dim, num_heads)

        self.use_adapter = use_adapter
        if self.use_adapter:
            self.adapters = nn.ModuleList([
                Adapter(projection_dim, bottleneck=16, dropout=dropout)
                for _ in range(num_layers)
            ])


    def forward(self,
                input_features: Tensor,
                input_points: Tensor,
                mask: Tensor,
                time: Tensor,
                pair_representation: Optional[Tensor] = None,
                pair_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor]=None,
                time_masking: Optional[Tensor]=None) -> tuple[Tensor, Optional[Tensor]]:
        """

        :param input_features: input features (batch_size, num_objects, num_features)
        :param input_points: subset of input features that used to do edge calculation (batch_size, num_objects, num_local_features)
        :param mask:  input features mask (batch_size, num_objects, 1)
        :param time: time input for diffusion model usage.
        :param time_masking: time masking for diffusion model usage (batch_size, num_objects, 1)
        :return:
        """
        if pair_representation is not None:
            if pair_representation.ndim != 4:
                raise ValueError(
                    "pair_representation must have shape [B, N, N, D_pair], "
                    f"got {tuple(pair_representation.shape)}."
                )
            expected_pair_shape = (
                input_features.shape[0],
                input_features.shape[1],
                input_features.shape[1],
            )
            if tuple(pair_representation.shape[:3]) != expected_pair_shape:
                raise ValueError(
                    "pair_representation must share [B, N, N] with input_features; "
                    f"expected {expected_pair_shape}, got {tuple(pair_representation.shape)}."
                )
            if (
                self.attention_bias_type is not None
                and pair_representation.shape[-1] != self.pair_input_dim
            ):
                raise ValueError(
                    f"Expected {self.pair_input_dim} input pair features, "
                    f"got {pair_representation.shape[-1]}."
                )
            if pair_mask is None:
                object_mask = mask.squeeze(-1).bool()
                pair_mask = object_mask[:, :, None] & object_mask[:, None, :]

        if self.attention_bias_type is not None and pair_representation is None:
            raise ValueError(
                f"attention_bias_type='{self.attention_bias_type}' requires pair_representation."
            )
        if pair_mask is not None:
            expected_mask_shape = (
                input_features.shape[0],
                input_features.shape[1],
                input_features.shape[1],
            )
            if tuple(pair_mask.shape) != expected_mask_shape:
                raise ValueError(
                    f"Expected pair_mask with shape {expected_mask_shape}, "
                    f"got {tuple(pair_mask.shape)}."
                )
            pair_mask = pair_mask.bool()
        if pair_representation is not None:
            # Keep returned pair state independent from the caller's input tensor.
            pair_representation = pair_representation.clone()

        encoded = self.random_drop(input_features)
        encoded = self.feature_embedding(encoded)

        static_attention_bias = None
        if self.attention_bias_type == "SimpleAddition":
            static_attention_bias = self.simple_pair_bias(pair_representation, pair_mask)
        elif self.attention_bias_type == "IterativeUpdate":
            pair_representation = self.pair_embedding(pair_representation, pair_mask)

        time = time.unsqueeze(1).unsqueeze(1).repeat(1, encoded.shape[1], 1)
        if time_masking is not None:
            time = time * time_masking # (batch_size, num_objects, 1)

        time_emb = self.time_embedding(time)
        time_emb = time_emb * mask

        time_emb = self.time_embed_linear(time_emb)
        scale, shift = torch.chunk(time_emb, 2, dim=-1)

        encoded = torch.add(torch.mul(encoded, (1.0 + scale)), shift)

        if self.use_local_embedding:
            points = input_points
            local_features = input_features
            local_features = self.local_embedding(
                x=local_features,
                points=points,
                mask=mask
            )
            encoded = local_features + encoded  # Combine with original features

        skip_connection = encoded
        for itransformer, transformer_block in enumerate(self.transformer_blocks):
            attention_bias = static_attention_bias
            if self.attention_bias_type == "IterativeUpdate":
                pair_representation = self.pair_update_blocks[itransformer](
                    pair_representation,
                    pair_mask,
                )
                attention_bias = self.iterative_pair_bias(pair_representation, pair_mask)

            encoded = transformer_block(
                x=encoded,
                mask=mask,
                attn_mask=attn_mask,
                attn_bias=attention_bias,
            )
            if self.use_adapter:
                encoded = self.adapters[itransformer](encoded)
                encoded = encoded * mask.float()


        return torch.add(encoded, skip_connection), pair_representation


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super(PositionEmbedding, self).__init__()

        self.position_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, current_embeddings: Tensor) -> Tensor:
        batch_size, num_vectors, input_dim = current_embeddings.shape

        position_embedding = self.position_embedding.expand(batch_size, num_vectors, -1)
        return torch.cat((current_embeddings, position_embedding), dim=2)


class CombinedEmbedding(nn.Module):
    def __init__(
            self,
            hidden_dim,
            position_embedding_dim
    ):
        super(CombinedEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.position_embedding_dim = position_embedding_dim

        self.first_position_embedding = PositionEmbedding(self.position_embedding_dim)
        self.second_position_embedding = PositionEmbedding(self.position_embedding_dim)
        self.final_embedding = create_linear_block(
            linear_block_type="GRU",
            input_dim=self.hidden_dim + self.position_embedding_dim,
            hidden_dim_scale=1.0,
            output_dim=self.hidden_dim,
            normalization_type="LayerNorm",
            activation_type="gelu",
            dropout=0.0,
            skip_connection=False
        )

    def forward(self, x: Tensor, y: Tensor, x_mask: Tensor, y_mask: Tensor) -> tuple[Any, Tensor]:
        """

        :param x: (batch_size, num_objects, hidden_dim)
        :param y: (batch_size, num_objects, hidden_dim)
        :param x_mask: (batch_size, num_objects, 1)
        :param y_mask: (batch_size, num_objects, 1)
        :return:
        """
        x_embed = self.first_position_embedding(x)
        y_embed = self.second_position_embedding(y)
        embeddings = torch.cat((x_embed, y_embed), dim=1)
        embeddings_mask = torch.cat((x_mask, y_mask), dim=1)
        embeddings = self.final_embedding(embeddings, embeddings_mask)
        return embeddings, embeddings_mask

class PointCloudPositionalEmbedding(nn.Module):
    def __init__(
            self,
            num_points: int,
            embed_dim: int
        ):
        super().__init__()
        self.position_embedding = nn.Embedding(num_points, embed_dim)

    def forward(self, x, time_mask, x_mask):
        # Generate position indices for each point in the sequence
        # x: [B, N ,D ]
        # time_mask: [ B, N ,1 ]
        # x_mask: [B, N, 1]

        time_mask = time_mask.squeeze(2)

        cumsum = time_mask.int().cumsum(dim=1)
        position_token = self.position_embedding(cumsum) # (B, N, D)
        position_token = position_token * time_mask.unsqueeze(-1)

        x = (x + position_token) * x_mask.float()

        return x  # (B, N, D)
