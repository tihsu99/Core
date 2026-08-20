import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evenet.dataset.types import InputType


class PairCreator(nn.Module):

    # Fixed numerical-safety thresholds, not model hyperparameters.
    EPS = 1e-8
    MAX_LOG1P_VALUE = 20.0
    MAX_ABS_ETA = 10.0
    MAX_ABS_PAIR_FEATURE = 30.0

    SUPPORTED = {
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
    }

    REQUIRED_INPUTS = {
        "deltaEta": {"eta"},
        "deltaPhi": {"phi"},
        "deltaR": {"eta", "phi"},
        "logDeltaR": {"eta", "phi"},
        "sinDeltaPhi": {"phi"},
        "cosDeltaPhi": {"phi"},
        "logMass": {"pt", "eta", "phi"},
        "Mass2": {"pt", "eta", "phi"},
        "logKT": {"pt", "eta", "phi"},
        "KT": {"pt", "eta", "phi"},
        "logPTRatio": {"pt"},
        "PTRatio": {"pt"},
    }

    def __init__(
        self,
        event_info,
        create: list[str]
    ):
        super().__init__()
        self.event_info = event_info
        self.create = tuple(create)

        if not self.create:
            raise ValueError("PairCreator requires at least one feature.")

        unknown = set(self.create) - self.SUPPORTED

        if unknown:
            raise ValueError(
                f"Unsupported PairCreator feature(s): {sorted(unknown)}. "
                f"Supported features are {sorted(self.SUPPORTED)}."
            )

        self.required_inputs = set()
        for name in self.create:
            self.required_inputs |= self.REQUIRED_INPUTS[name]

        # --------------------------------------------------------
        # Resolve INPUTS.SEQUENTIAL feature name -> tensor column.
        # --------------------------------------------------------

        self.feature_indices = {}
        self.feature_info = {}

        duplicate_features = set()
        sequential_index = 0

        for input_name, features in event_info.input_features.items():
            if event_info.input_types[input_name] != InputType.Sequential:
                continue
            for feature in features:
                if feature.name in self.feature_indices:
                    duplicate_features.add(feature.name)
                else:
                    self.feature_indices[feature.name] = sequential_index
                    self.feature_info[feature.name] = feature
                sequential_index += 1

        ambiguous = duplicate_features & self.required_inputs
        if ambiguous:
            raise ValueError(
                "PairCreator requires ambiguous SEQUENTIAL feature names: "
                f"{sorted(ambiguous)}"
            )

        missing = self.required_inputs - set(self.feature_indices)

        if missing:
            raise ValueError(
                "PairCreator requires missing INPUTS.SEQUENTIAL features: "
                f"{sorted(missing)}"
            )

    @property
    def output_dim(self) -> int:
        return len(self.create)

    def _read_physical_feature(
        self,
        x: Tensor,
        name: str
    ) -> Tensor:
        """
        x is PRE-NORMALIZATION EveNet input.
        Important:
            - log_scale variables have already undergone log1p
                during preprocessing.
            - standard normalization has NOT been applied yet here.
        """

        index = self.feature_indices[name]
        info = self.feature_info[name]

        value = x[..., index].float()

        # First remove any upstream non-finite values.

        value = torch.nan_to_num(
            value,
            nan=0.0,
            posinf=self.MAX_LOG1P_VALUE if info.log_scale else 0.0,
            neginf=0.0
        )

        if info.log_scale:
            value = torch.clamp(
                value,
                min=0.0,
                max=self.MAX_LOG1P_VALUE
            )
            value = torch.expm1(value)

        return value

    @staticmethod
    def _wrap_delta_phi(delta_phi: Tensor)->Tensor:
        """
        Wrap delta_phi to [-pi, pi).
        """
        return (delta_phi + torch.pi) % (2 * torch.pi) - torch.pi

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        output_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : PRE-NORMALIZATION sequential input
            [B, N, F_input].
        mask:
            [B, N] or [B, N, 1]
        Returns
        -----------
        pair_features : [B, N, N, F_pair]
            Pairwise features for each particle pair.
        pair_mask : [B, N, N]
            Mask indicating valid particle pairs.
        """

        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, N, F], got {tuple(x.shape)}.")
        if mask.ndim == 3 and mask.shape[-1] != 1:
            raise ValueError(f"Expected mask with shape [B, N] or [B, N, 1], got {tuple(mask.shape)}.")
        if mask.ndim not in (2, 3):
            raise ValueError(f"Expected mask with shape [B, N] or [B, N, 1], got {tuple(mask.shape)}.")
        if mask.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"x and mask must share [B, N], got {tuple(x.shape)} and {tuple(mask.shape)}."
            )

        if mask.ndim == 3:
            valid = mask.squeeze(-1).bool()
        else:
            valid = mask.bool()

        pair_mask = (
            valid[:, :, None] & valid[:, None, :]
        )

        feature_bank = {}

        # ---------------------
        # Recover only physics variables actually needed
        # ---------------------

        pt = None
        eta = None
        phi = None

        if "pt" in self.required_inputs:
            pt = self._read_physical_feature(x, "pt")
        if "eta" in self.required_inputs:
            eta = self._read_physical_feature(x, "eta")
        if "phi" in self.required_inputs:
            phi = self._read_physical_feature(x, "phi")
            phi = self._wrap_delta_phi(phi)

        delta_eta = None
        delta_phi = None
        delta_r = None

        if eta is not None and "deltaEta" in self.create:
            delta_eta = (
                    eta[:, :, None]
                    - eta[:, None, :]
            )
            feature_bank["deltaEta"] = delta_eta

        if phi is not None and (
            "deltaPhi" in self.create
            or "sinDeltaPhi" in self.create
            or "cosDeltaPhi" in self.create
        ):
            delta_phi = (
                    phi[:, :, None]
                    - phi[:, None, :]
            )
            delta_phi = self._wrap_delta_phi(delta_phi)
            if "deltaPhi" in self.create:
                feature_bank["deltaPhi"] = delta_phi
            if "sinDeltaPhi" in self.create:
                feature_bank["sinDeltaPhi"] = torch.sin(delta_phi)
            if "cosDeltaPhi" in self.create:
                feature_bank["cosDeltaPhi"] = torch.cos(delta_phi)

        if eta is not None and phi is not None and ("deltaR" in self.create or "logDeltaR" in self.create):
            if delta_eta is None:
                delta_eta = (
                    eta[:, :, None]
                    - eta[:, None, :]
                )
            if delta_phi is None:
                delta_phi = (
                    phi[:, :, None]
                    - phi[:, None, :]
                )
                delta_phi = self._wrap_delta_phi(delta_phi)
            delta_r2 = delta_eta ** 2 + delta_phi ** 2
            delta_r2 = torch.clamp(delta_r2, min=0.0)
            delta_r = torch.sqrt(delta_r2)

            if "logDeltaR" in self.create:
                feature_bank["logDeltaR"] = torch.log1p(delta_r)
            if "deltaR" in self.create:
                feature_bank["deltaR"] = delta_r

        # ---------------------
        # pT pair quantities
        # ---------------------

        pt_i = None
        pt_j = None
        log_pt_ratio = None
        if pt is not None and ("logPTRatio" in self.create or "PTRatio" in self.create):
            pt_i = pt[:, :, None]
            pt_j = pt[:, None, :]
            pt_i = torch.clamp(pt_i, min=self.EPS)
            pt_j = torch.clamp(pt_j, min=self.EPS)
            log_pt_ratio = torch.log(pt_i) - torch.log(pt_j)
            if "PTRatio" in self.create:
                feature_bank["PTRatio"] = pt_i / pt_j
            if "logPTRatio" in self.create:
                feature_bank["logPTRatio"] = log_pt_ratio

        # ---------------------
        # Massless pair invariant mass:
        # m_ij^2 = 2 pt_i pt_j [cosh(delta_eta) - cos(delta_phi)].
        # This stays meaningful for independently noised diffusion features.
        # ---------------------

        pair_mass = None

        if (
            ("logMass" in self.create or "Mass2" in self.create)
            and pt is not None
            and eta is not None
            and phi is not None
        ):
            if delta_eta is None:
                delta_eta = eta[:, :, None] - eta[:, None, :]
            if delta_phi is None:
                delta_phi = self._wrap_delta_phi(
                    phi[:, :, None] - phi[:, None, :]
                )

            safe_delta_eta = torch.clamp(
                delta_eta,
                min=-self.MAX_ABS_ETA,
                max=self.MAX_ABS_ETA,
            )
            mass2 = (
                2.0
                * pt[:, :, None]
                * pt[:, None, :]
                * (torch.cosh(safe_delta_eta) - torch.cos(delta_phi))
            )

            mass2 = torch.nan_to_num(
                mass2,
                nan=0.0,
                posinf=torch.finfo(mass2.dtype).max,
                neginf=0.0,
            )
            mass2 = torch.clamp(mass2, min=0.0)

            pair_mass = torch.sqrt(mass2)
            if "logMass" in self.create:
                feature_bank["logMass"] = torch.log1p(pair_mass)
            if "Mass2" in self.create:
                feature_bank["Mass2"] = mass2
        # -------------------------------------------------------------
        # kT
        #
        # kT_ij = min(pt_i, pt_j) * deltaR_ij
        # -------------------------------------------------------------

        pair_kt = None

        if (
            ("logKT" in self.create or "KT" in self.create)
            and pt is not None
            and eta is not None
            and phi is not None
        ):
            if delta_r is None:
                delta_eta = (
                    eta[:, :, None]
                    - eta[:, None, :]
                )
                delta_phi = (
                    phi[:, :, None]
                    - phi[:, None, :]
                )
                delta_phi = self._wrap_delta_phi(delta_phi)
                delta_r2 = delta_eta ** 2 + delta_phi ** 2
                delta_r2 = torch.clamp(delta_r2, min=0.0)
                delta_r = torch.sqrt(delta_r2)

            pair_kt = (
                torch.minimum(pt[:, :, None], pt[:, None, :])
                * delta_r
            )

            pair_kt = torch.nan_to_num(
                pair_kt,
                nan=0.0,
                posinf=torch.finfo(pair_kt.dtype).max,
                neginf=0.0,
            )

            if "KT" in self.create:
                feature_bank["KT"] = pair_kt
            if "logKT" in self.create:
                feature_bank["logKT"] = torch.log1p(pair_kt)

        # -------------------------------------------------------------
        # Follow YAML ordering exactly.
        # -------------------------------------------------------------

        pair_features = torch.stack(
            [feature_bank[name] for name in self.create],
            dim=-1,
        )
        # -------------------------------------------------------------
        # FINAL SAFETY NET
        #
        # Important:
        # Do this BEFORE multiplying by the mask.
        #
        # NaN * 0 is still NaN.
        # -------------------------------------------------------------

        pair_features = torch.nan_to_num(
            pair_features,
            nan=0.0,
            posinf=self.MAX_ABS_PAIR_FEATURE,
            neginf=-self.MAX_ABS_PAIR_FEATURE,
        )

        pair_features = torch.clamp(
            pair_features,
            min=-self.MAX_ABS_PAIR_FEATURE,
            max=self.MAX_ABS_PAIR_FEATURE,
        )

        n = valid.shape[1]

        diagonal = torch.eye(
            n,
            dtype=torch.bool,
            device=valid.device,
        )[None, :, :]

        pair_feature_mask = pair_mask & ~diagonal

        pair_features = torch.where(
            pair_feature_mask[..., None],
            pair_features,
            torch.zeros_like(pair_features),
        )

        if output_size is not None:
            padding = output_size - n
            if padding < 0:
                raise ValueError(
                    f"Pair output size {output_size} is smaller than input size {n}."
                )
            if padding:
                pair_features = F.pad(pair_features, (0, 0, 0, padding, 0, padding))
                pair_mask = F.pad(pair_mask, (0, padding, 0, padding), value=False)

        return pair_features, pair_mask
