from gridfm_graphkit.datasets.globals import *
from gridfm_graphkit.io.registries import NORMALIZERS_REGISTRY
import torch
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData


class Normalizer(ABC):
    """
    Abstract base class for all normalization strategies.
    """

    @abstractmethod
    def fit(self, data: torch.Tensor) -> dict:
        """
        Fit normalization parameters from data.

        Args:
            data: Input tensor.

        Returns:
            Dictionary of computed parameters.
        """

    @abstractmethod
    def fit_from_dict(self, params: dict):
        """
        Set parameters from a precomputed dictionary.

        Args:
            params: Dictionary of parameters.
        """

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input data.

        Args:
            data: Input tensor.

        Returns:
            Normalized tensor.
        """

    @abstractmethod
    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Undo normalization.

        Args:
            normalized_data: Normalized tensor.

        Returns:
            Original tensor.
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Return the stored normalization statistics for logging/inspection.
        """


@NORMALIZERS_REGISTRY.register("HeteroDataMVANormalizer")
class HeteroDataMVANormalizer(Normalizer):
    """
    In power systems, a suitable normalization strategy must preserve the physical properties of
    the system. A known method is the conversion to the per-unit (p.u.) system, which expresses
    electrical quantities such as voltage, current, power, and impedance as fractions of predefined
    base values. These base values are usually chosen based on system parameters, such as rated
    voltage. The per-unit conversion ensures that power system equations remain scale-invariant,
    preserving fundamental physical relationships.
    """

    def __init__(self, args):
        """
        Args:
            node_data: Whether data is node-level or edge-level
            args (NestedNamespace): Parameters

        Attributes:
            baseMVA (float): baseMVA found in casefile. From ``args.data.baseMVA``.
        """
        self.baseMVA_orig = getattr(args.data, "baseMVA", 100)
        self.baseMVA = None

    def to(self, device):
        pass

    def fit(
        self,
        bus_data: Optional[pd.DataFrame] = None,
        gen_data: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Fit method for both node-level and edge-level normalization.

        - For node-level: pass `bus_data` and `gen_data`.
        - For edge-level: pass `baseMVA` directly.
        """
        if bus_data is None or gen_data is None:
            raise ValueError(
                "bus_data and gen_data must be provided for node-level normalization.",
            )

        pd_values = bus_data["Pd"]
        qd_values = bus_data["Qd"]
        pg_values = gen_data["p_mw"]
        qg_values = bus_data["Qg"]

        non_zero_values = pd.concat(
            [
                pd_values[pd_values != 0],
                qd_values[qd_values != 0],
                pg_values[pg_values != 0],
                qg_values[qg_values != 0],
            ],
        )

        self.baseMVA = np.percentile(non_zero_values, 95)
        self.vn_kv_max = float(bus_data["vn_kv"].max())

        # ----------------------------------------------------
        # Return all fitted parameters
        # ----------------------------------------------------
        return {
            "baseMVA_orig": self.baseMVA_orig,
            "baseMVA": self.baseMVA,
            "vn_kv_max": self.vn_kv_max,
        }

    def fit_from_dict(self, params: dict):
        # Base MVA
        self.baseMVA = params.get("baseMVA")

        self.baseMVA_orig = params.get("baseMVA_orig")

        # vn_kv
        self.vn_kv_max = params.get("vn_kv_max")

    def transform(self, data: HeteroData):
        if self.baseMVA is None or self.baseMVA == 0:
            raise ValueError("BaseMVA not properly set")

        # --- Bus input normalization ---
        data.x_dict["bus"][:, PD_H] /= self.baseMVA
        data.x_dict["bus"][:, QD_H] /= self.baseMVA
        data.x_dict["bus"][:, QG_H] /= self.baseMVA
        data.x_dict["bus"][:, MIN_QG_H] /= self.baseMVA
        data.x_dict["bus"][:, MAX_QG_H] /= self.baseMVA
        data.x_dict["bus"][:, VA_H] *= torch.pi / 180.0
        data.x_dict["bus"][:, GS] *= self.baseMVA_orig / self.baseMVA
        data.x_dict["bus"][:, BS] *= self.baseMVA_orig / self.baseMVA
        data.x_dict["bus"][:, VN_KV] /= self.vn_kv_max

        # --- Bus label normalization ---
        data.y_dict["bus"][:, PD_H] /= self.baseMVA
        data.y_dict["bus"][:, QD_H] /= self.baseMVA
        data.y_dict["bus"][:, QG_H] /= self.baseMVA
        data.y_dict["bus"][:, VA_H] *= torch.pi / 180.0

        # --- Generator input normalization ---
        data.x_dict["gen"][:, PG_H] /= self.baseMVA
        data.x_dict["gen"][:, MIN_PG] /= self.baseMVA
        data.x_dict["gen"][:, MAX_PG] /= self.baseMVA
        data.x_dict["gen"][:, C0_H] = torch.sign(
            data.x_dict["gen"][:, C0_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C0_H]))
        data.x_dict["gen"][:, C1_H] = torch.sign(
            data.x_dict["gen"][:, C1_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C1_H]))
        data.x_dict["gen"][:, C2_H] = torch.sign(
            data.x_dict["gen"][:, C2_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C2_H]))

        # --- Generator label normalization ---
        data.y_dict["gen"][:, PG_H] /= self.baseMVA

        # --- Edge input normalization ---
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] /= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] /= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            self.baseMVA_orig / self.baseMVA
        )
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] /= self.baseMVA

    def inverse_transform(self, data: HeteroData):
        if self.baseMVA is None or self.baseMVA == 0:
            raise ValueError("BaseMVA not properly set")

        # -------- BUS INPUT INVERSE NORMALIZATION --------
        data.x_dict["bus"][:, PD_H] *= self.baseMVA
        data.x_dict["bus"][:, QD_H] *= self.baseMVA
        data.x_dict["bus"][:, QG_H] *= self.baseMVA
        data.x_dict["bus"][:, MIN_QG_H] *= self.baseMVA
        data.x_dict["bus"][:, MAX_QG_H] *= self.baseMVA
        data.x_dict["bus"][:, GS] *= self.baseMVA
        data.x_dict["bus"][:, BS] *= self.baseMVA
        data.x_dict["bus"][:, VN_KV] *= self.vn_kv_max

        # -------- BUS LABEL INVERSE NORMALIZATION --------
        data.y_dict["bus"][:, PD_H] *= self.baseMVA
        data.y_dict["bus"][:, QD_H] *= self.baseMVA
        data.y_dict["bus"][:, QG_H] *= self.baseMVA

        # -------- GENERATOR INPUT INVERSE NORMALIZATION --------
        data.x_dict["gen"][:, PG_H] *= self.baseMVA
        data.x_dict["gen"][:, MIN_PG] *= self.baseMVA
        data.x_dict["gen"][:, MAX_PG] *= self.baseMVA
        data.x_dict["gen"][:, C0_H] = torch.sign(data.x_dict["gen"][:, C0_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C0_H])) - 1
        )
        data.x_dict["gen"][:, C1_H] = torch.sign(data.x_dict["gen"][:, C1_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C1_H])) - 1
        )
        data.x_dict["gen"][:, C2_H] = torch.sign(data.x_dict["gen"][:, C2_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C2_H])) - 1
        )

        # -------- GENERATOR LABEL INVERSE NORMALIZATION --------
        data.y_dict["gen"][:, PG_H] *= self.baseMVA

        # -------- EDGE INPUT INVERSE NORMALIZATION --------
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] *= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] *= self.baseMVA

        # Inverse of scaling Y and tap parameters
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            self.baseMVA
        )

        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= 180.0 / torch.pi
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= 180.0 / torch.pi

        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] *= self.baseMVA

    def inverse_output(self, output):
        bus_output = output["bus"]
        gen_output = output["gen"]
        bus_output[:, PG_OUT] *= self.baseMVA
        bus_output[:, QG_OUT] *= self.baseMVA
        gen_output[:, PG_H] *= self.baseMVA

    def get_stats(self) -> dict:
        return {
            "baseMVA": self.baseMVA,
            "baseMVA_orig": self.baseMVA_orig,
            "vn_kv_max": self.vn_kv_max,
        }
