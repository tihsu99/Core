import torch

from evenet.network.body.embedding import LocalEmbeddingLayer, PETBody
from evenet.network.body.pairformer import ObjectToPair
from evenet.network.layers.norm import DynamicTanh
from evenet.network.layers.transformer import TransformerBlockModule


class RecordingAttention(torch.nn.Module):
    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
    ):
        self.key_padding_mask = key_padding_mask
        self.attn_mask = attn_mask
        return torch.zeros_like(query), None


def make_pet(
    attention_bias_type=None,
    *,
    local=True,
    k=0,
    use_triangle_attention=False,
    use_object_to_pair=False,
    talking_head=False,
    norm_type="DynamicTanh",
):
    return PETBody(
        num_feat=4,
        num_keep=0,
        feature_drop=0.0,
        projection_dim=8,
        local=local,
        K=k,
        num_local=2,
        num_layers=2,
        num_heads=2,
        drop_probability=0.0,
        talking_head=talking_head,
        layer_scale=False,
        layer_scale_init=1.0e-5,
        dropout=0.0,
        mode="all",
        attention_bias_type=attention_bias_type,
        pair_input_dim=3 if attention_bias_type is not None else 0,
        pair_dim=8,
        pair_num_heads=2,
        use_triangle_attention=use_triangle_attention,
        use_object_to_pair=use_object_to_pair,
        norm_type=norm_type,
    ).eval()


def make_inputs():
    torch.manual_seed(7)
    features = torch.randn(2, 4, 4)
    mask = torch.tensor([
        [[1.0], [1.0], [1.0], [0.0]],
        [[1.0], [1.0], [0.0], [0.0]],
    ])
    valid = mask.squeeze(-1).bool()
    pair_mask = valid[:, :, None] & valid[:, None, :]
    pair = torch.randn(2, 4, 4, 3) * pair_mask.unsqueeze(-1)
    time = torch.zeros(2)
    return features, mask, pair, pair_mask, time


def test_k_zero_bypasses_local_knn():
    model = make_pet(local=True, k=0)
    features, mask, _, _, time = make_inputs()

    output, pair_output, pair_input = model(
        input_features=features,
        input_points=features[..., :2],
        mask=mask,
        time=time,
    )

    assert not model.use_local_embedding
    assert output.shape == (2, 4, 8)
    assert pair_output is None
    assert pair_input is None
    assert torch.isfinite(output).all()


def test_local_embedding_uses_fixed_masked_physical_knn():
    layer = LocalEmbeddingLayer(
        input_dim=4,
        projection_dim=8,
        K=3,
        num_local=1,
        num_heads=2,
        dropout=0.0,
        norm_type="DynamicTanh",
    ).eval()
    features = torch.randn(1, 4, 4)
    changed_features = features * 100.0
    points = torch.tensor([[[0.0], [1.0], [10.0], [999.0]]])
    mask = torch.tensor([[[True], [True], [True], [False]]])

    output, indices, neighbor_mask = layer(features, points, mask)
    changed_output, changed_indices, changed_neighbor_mask = layer(
        changed_features, points, mask
    )

    assert torch.equal(indices, changed_indices)
    assert torch.equal(neighbor_mask, changed_neighbor_mask)
    assert torch.equal(neighbor_mask.sum(dim=-1), torch.tensor([[2, 2, 2, 0]]))
    assert torch.all(output[:, 3] == 0)
    assert torch.isfinite(output).all()
    assert not torch.allclose(output[:, :3], changed_output[:, :3])


def test_simple_addition_uses_static_pair_bias_and_returns_independent_pair_output():
    model = make_pet("SimpleAddition")
    features, mask, pair, pair_mask, time = make_inputs()

    output, pair_output, pair_input = model(
        input_features=features,
        input_points=features[..., :2],
        mask=mask,
        time=time,
        pair_representation=pair,
        pair_mask=pair_mask,
    )
    output_without_pair_signal, _, _ = model(
        input_features=features,
        input_points=features[..., :2],
        mask=mask,
        time=time,
        pair_representation=torch.zeros_like(pair),
        pair_mask=pair_mask,
    )

    assert pair_output.shape == pair.shape
    assert pair_input is None
    assert pair_output.data_ptr() != pair.data_ptr()
    assert torch.equal(pair_output, pair)
    assert not torch.allclose(output, output_without_pair_signal)


def test_iterative_update_returns_masked_persistent_pair_state_without_triangle_attention():
    model = make_pet("IterativeUpdate", use_triangle_attention=False)
    features, mask, pair, pair_mask, time = make_inputs()
    hard_attention_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    hard_attention_mask[:, 0, 2] = True

    output, pair_output, pair_input = model(
        input_features=features,
        input_points=features[..., :2],
        mask=mask,
        time=time,
        pair_representation=pair,
        pair_mask=pair_mask,
        attn_mask=hard_attention_mask,
    )

    assert not model.use_triangle_attention
    assert pair_output.shape == (2, 4, 4, 8)
    assert pair_input is None
    assert torch.isfinite(output).all()
    assert torch.isfinite(pair_output).all()
    assert torch.all(pair_output[~pair_mask] == 0)
    assert all(
        not block.use_triangle_attention
        for block in model.pair_update_blocks
    )


def test_iterative_update_exposes_p0_only_when_requested():
    model = make_pet("IterativeUpdate", use_triangle_attention=False)
    features, mask, pair, pair_mask, time = make_inputs()

    output, pl, p0 = model(
        input_features=features,
        input_points=features[..., :2],
        mask=mask,
        time=time,
        pair_representation=pair,
        pair_mask=pair_mask,
        return_pair_states=True,
    )

    assert output.shape == (2, 4, 8)
    assert p0.shape == pl.shape == (2, 4, 4, 8)
    expected_p0 = model.pair_embedding(pair, pair_mask)
    assert torch.allclose(p0, expected_p0)
    assert torch.all(p0[~pair_mask] == 0)
    assert torch.all(pl[~pair_mask] == 0)
    assert p0.data_ptr() != pl.data_ptr()


def test_ordered_object_latents_update_pair_state():
    model = make_pet("IterativeUpdate", use_object_to_pair=True)
    features, mask, pair, pair_mask, time = make_inputs()

    output, pair_output, _ = model(
        input_features=features,
        input_points=features[..., :2],
        mask=mask,
        time=time,
        pair_representation=pair,
        pair_mask=pair_mask,
    )
    _, changed_pair_output, _ = model(
        input_features=features.flip(1),
        input_points=features.flip(1)[..., :2],
        mask=mask,
        time=time,
        pair_representation=pair,
        pair_mask=pair_mask,
    )

    assert model.use_object_to_pair
    assert not torch.allclose(pair_output, changed_pair_output)
    assert not torch.allclose(pair_output[:, 0, 1], pair_output[:, 1, 0])
    assert torch.all(pair_output[~pair_mask] == 0)
    object_to_pair_params = [
        parameter
        for block in model.object_to_pair_blocks
        for parameter in block.parameters()
    ]
    grads = torch.autograd.grad(output.square().sum(), object_to_pair_params)
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_object_to_pair_keeps_source_and_target_order():
    layer = ObjectToPair(object_dim=2, pair_dim=1).eval()
    with torch.no_grad():
        layer.projection[1].weight.zero_()
        layer.projection[1].bias.zero_()
        layer.projection[1].weight[0, 0] = 1.0
        layer.projection[3].weight.zero_()
        layer.projection[3].bias.zero_()
        layer.projection[3].weight[0, 0] = 1.0

    objects = torch.tensor([[[3.0, 0.0], [0.0, 1.0], [9.0, 9.0]]])
    pair_mask = torch.tensor([[
        [True, True, False],
        [True, True, False],
        [False, False, False],
    ]])
    update = layer(objects, pair_mask)

    assert not torch.allclose(update[:, 0, 1], update[:, 1, 0])
    assert torch.all(update[~pair_mask] == 0)


def test_triangle_attention_is_explicitly_opt_in():
    model = make_pet("IterativeUpdate", use_triangle_attention=True)
    features, mask, pair, pair_mask, time = make_inputs()

    output, pair_output, pair_input = model(
        input_features=features,
        input_points=features[..., :2],
        mask=mask,
        time=time,
        pair_representation=pair,
        pair_mask=pair_mask,
    )

    assert model.use_triangle_attention
    assert pair_input is None
    assert all(block.use_triangle_attention for block in model.pair_update_blocks)
    assert torch.isfinite(output).all()
    assert torch.isfinite(pair_output).all()


def test_pair_bias_also_works_with_talking_head_attention():
    model = make_pet("SimpleAddition", talking_head=True)
    features, mask, pair, pair_mask, time = make_inputs()

    output, _, _ = model(
        input_features=features,
        input_points=features[..., :2],
        mask=mask,
        time=time,
        pair_representation=pair,
        pair_mask=pair_mask,
    )

    assert torch.isfinite(output).all()


def test_attention_mask_sign_and_validity_conventions_are_not_flipped():
    block = TransformerBlockModule(
        projection_dim=8,
        num_heads=2,
        dropout=0.0,
        talking_head=False,
        layer_scale=False,
        layer_scale_init=1.0e-5,
        drop_probability=0.0,
    ).eval()
    recorder = RecordingAttention()
    block.attn = recorder

    x = torch.randn(1, 3, 8)
    valid_object_mask = torch.tensor([[[True], [False], [True]]])
    pair_bias = torch.zeros(1, 2, 3, 3)
    pair_bias[:, :, 0, 0] = 2.0
    pair_bias[:, :, 0, 2] = -3.0
    blocked_attention_mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    blocked_attention_mask[:, 2, 0] = True

    block(
        x=x,
        mask=valid_object_mask,
        attn_bias=pair_bias,
        attn_mask=blocked_attention_mask,
    )

    additive_mask = recorder.attn_mask.view(1, 2, 3, 3)
    blocked_value = torch.finfo(additive_mask.dtype).min

    # Positive and negative learned biases retain their original meaning.
    assert torch.all(additive_mask[:, :, 0, 0] == 2.0)
    assert torch.all(additive_mask[:, :, 0, 2] == -3.0)
    # Boolean True blocks exactly the requested query-to-key edge.
    assert torch.all(additive_mask[:, :, 2, 0] == blocked_value)
    # Invalid object 1 is masked only as a key column; valid keys remain usable.
    assert torch.all(additive_mask[:, :, :, 1] == blocked_value)
    assert recorder.key_padding_mask is None

    float_attention_mask = torch.zeros(3, 3)
    float_attention_mask[1, 0] = 1.5
    float_attention_mask[1, 2] = -2.5
    prepared = block._prepare_attention_bias(
        x=x,
        attn_mask=float_attention_mask,
    )
    assert torch.all(prepared[:, :, 1, 0] == 1.5)
    assert torch.all(prepared[:, :, 1, 2] == -2.5)


def test_dynamic_tanh_is_a_pet_only_configurable_normalization():
    layer = DynamicTanh(normalized_shape=3)
    x = torch.tensor([[[2.0, -1.0, 0.5]]], requires_grad=True)
    expected = layer.weight * torch.tanh(layer.alpha * x)
    output = layer(x)
    assert torch.equal(output, expected)

    model = make_pet()
    assert all(
        isinstance(block.norm1, DynamicTanh)
        and isinstance(block.norm2, DynamicTanh)
        for block in model.transformer_blocks
    )

    features, mask, _, _, time = make_inputs()
    pet_output, _, _ = model(
        input_features=features,
        input_points=features[..., :2],
        mask=mask,
        time=time,
    )
    pet_output.square().mean().backward()
    assert torch.isfinite(pet_output).all()
    assert all(block.norm1.alpha.grad is not None for block in model.transformer_blocks)

    legacy_model = make_pet(norm_type="LayerNorm")
    assert all(
        isinstance(block.norm1, torch.nn.LayerNorm)
        and isinstance(block.norm2, torch.nn.LayerNorm)
        for block in legacy_model.transformer_blocks
    )
    legacy_keys = legacy_model.transformer_blocks[0].state_dict()
    assert "norm1.weight" in legacy_keys
    assert "norm1.bias" in legacy_keys
    assert "norm2.weight" in legacy_keys
    assert "norm2.bias" in legacy_keys


if __name__ == "__main__":
    test_k_zero_bypasses_local_knn()
    test_local_embedding_uses_fixed_masked_physical_knn()
    test_simple_addition_uses_static_pair_bias_and_returns_independent_pair_output()
    test_iterative_update_returns_masked_persistent_pair_state_without_triangle_attention()
    test_iterative_update_exposes_p0_only_when_requested()
    test_ordered_object_latents_update_pair_state()
    test_object_to_pair_keeps_source_and_target_order()
    test_triangle_attention_is_explicitly_opt_in()
    test_pair_bias_also_works_with_talking_head_attention()
    test_attention_mask_sign_and_validity_conventions_are_not_flipped()
    test_dynamic_tanh_is_a_pet_only_configurable_normalization()
