from collections import OrderedDict
from types import SimpleNamespace

import torch

from evenet.dataset.types import FeatureInfo, InputType
from evenet.network.body.pair_creator import PairCreator


def test_pair_creator_is_finite_and_excludes_masked_objects():
    event_info = SimpleNamespace(
        input_types=OrderedDict([("objects", InputType.Sequential)]),
        input_features=OrderedDict([
            ("objects", (
                FeatureInfo("energy", True, True, False),
                FeatureInfo("pt", True, True, False),
                FeatureInfo("eta", True, False, False),
                FeatureInfo("phi", True, False, True),
            )),
        ]),
    )
    names = (
        "deltaEta",
        "deltaPhi",
        "deltaR",
        "logDeltaR",
        "sinDeltaPhi",
        "cosDeltaPhi",
        "logMass",
        "Mass2",
        "logKT",
        "KT",
        "logPTRatio",
        "PTRatio",
    )
    creator = PairCreator(event_info=event_info, create=list(names))

    # The third object represents padding (including future projected neutrino slots).
    x = torch.tensor([[[
        torch.log1p(torch.tensor(50.0)),
        torch.log1p(torch.tensor(40.0)),
        0.2,
        torch.pi - 0.01,
    ], [
        torch.log1p(torch.tensor(35.0)),
        torch.log1p(torch.tensor(30.0)),
        -0.3,
        -torch.pi + 0.01,
    ], [float("nan"), float("inf"), 1.0e20, float("-inf")]]])
    mask = torch.tensor([[[True], [True], [False]]])

    pair, pair_mask = creator(x, mask, output_size=5)
    index = {name: names.index(name) for name in names}

    assert pair.shape == (1, 5, 5, len(names))
    assert pair_mask.shape == (1, 5, 5)
    assert torch.isfinite(pair).all()
    assert torch.all(pair[:, 2:] == 0)
    assert torch.all(pair[:, :, 2:] == 0)
    assert not pair_mask[:, 2:].any()
    assert not pair_mask[:, :, 2:].any()
    assert torch.all(pair.diagonal(dim1=1, dim2=2) == 0)

    assert torch.allclose(
        pair[..., index["deltaR"]],
        pair[..., index["deltaR"]].transpose(1, 2),
    )
    assert torch.allclose(
        pair[..., index["logMass"]],
        pair[..., index["logMass"]].transpose(1, 2),
    )
    assert torch.allclose(
        pair[..., index["deltaEta"]],
        -pair[..., index["deltaEta"]].transpose(1, 2),
    )
    assert torch.allclose(
        pair[..., index["logPTRatio"]],
        -pair[..., index["logPTRatio"]].transpose(1, 2),
    )
    assert torch.allclose(
        pair[0, 0, 1, index["deltaPhi"]].abs(),
        torch.tensor(0.02),
        atol=1e-5,
    )
if __name__ == "__main__":
    test_pair_creator_is_finite_and_excludes_masked_objects()
