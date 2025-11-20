from gridfm_graphkit.datasets.globals import *
from gridfm_graphkit.io.registries import NORMALIZERS_REGISTRY
import torch
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import pandas as pd
import numpy as np


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


@NORMALIZERS_REGISTRY.register("minmax")
class MinMaxNormalizer(Normalizer):
    """
    Scales each feature to the [0, 1] range.

    Args:
        node_data (bool): Whether data is node-level or edge-level
        args (NestedNamespace): Parameters

    """

    def __init__(self, node_data: bool, args):
        self.min_val = None
        self.max_val = None

    def to(self, device):
        self.min_val = self.min_val.to(device)
        self.max_val = self.max_val.to(device)

    def fit(self, data: torch.Tensor) -> dict:
        self.min_val, _ = data.min(axis=0)
        self.max_val, _ = data.max(axis=0)

        return {"min_value": self.min_val, "max_value": self.max_val}

    def fit_from_dict(self, params: dict):
        if self.min_val is None:
            self.min_val = params.get("min_value")
        if self.max_val is None:
            self.max_val = params.get("max_value")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.min_val is None or self.max_val is None:
            raise ValueError("fit must be called before transform.")

        diff = self.max_val - self.min_val
        diff[diff == 0] = 1  # Avoid division by zero for features with zero range
        return (data - self.min_val) / diff

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        if self.min_val is None or self.max_val is None:
            raise ValueError("fit must be called before inverse_transform.")

        diff = self.max_val - self.min_val
        diff[diff == 0] = 1
        return (normalized_data * diff) + self.min_val

    def get_stats(self) -> dict:
        return {
            "min_value": self.min_val.tolist() if self.min_val is not None else None,
            "max_value": self.max_val.tolist() if self.max_val is not None else None,
        }


@NORMALIZERS_REGISTRY.register("standard")
class Standardizer(Normalizer):
    """
    Standardizes each feature to zero mean and unit variance.

    Args:
        node_data (bool): Whether data is node-level or edge-level
        args (NestedNamespace): Parameters

    """

    def __init__(self, node_data: bool, args):
        self.mean = None
        self.std = None

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def fit(self, data: torch.Tensor) -> dict:
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

        return {"mean_value": self.mean, "std_value": self.std}

    def fit_from_dict(self, params: dict):
        if self.mean is None:
            self.mean = params.get("mean_value")
        if self.std is None:
            self.std = params.get("std_value")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("fit must be called before transform.")

        std = self.std.clone()
        std[std == 0] = 1  # Avoid division by zero for features with zero std
        return (data - self.mean) / std

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("fit must be called before inverse_transform.")

        std = self.std.clone()
        std[std == 0] = 1
        return (normalized_data * std) + self.mean

    def get_stats(self) -> dict:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }


@NORMALIZERS_REGISTRY.register("baseMVAnorm")
class BaseMVANormalizer(Normalizer):
    """
    In power systems, a suitable normalization strategy must preserve the physical properties of
    the system. A known method is the conversion to the per-unit (p.u.) system, which expresses
    electrical quantities such as voltage, current, power, and impedance as fractions of predefined
    base values. These base values are usually chosen based on system parameters, such as rated
    voltage. The per-unit conversion ensures that power system equations remain scale-invariant,
    preserving fundamental physical relationships.
    """

    def __init__(self, node_data: bool, args):
        """
        Args:
            node_data: Whether data is node-level or edge-level
            args (NestedNamespace): Parameters

        Attributes:
            baseMVA (float): baseMVA found in casefile. From ``args.data.baseMVA``.
        """
        self.node_data = node_data
        self.baseMVA_orig = getattr(args.data, "baseMVA", 100)
        self.baseMVA = None

    def to(self, device):
        pass

    def fit(self, data: torch.Tensor, baseMVA: float = None) -> dict:
        if self.node_data:
            self.baseMVA = data[:, [PD, QD, PG, QG]].max()
        else:
            self.baseMVA = baseMVA

        return {"baseMVA_orig": self.baseMVA_orig, "baseMVA": self.baseMVA}

    def fit_from_dict(self, params: dict):
        if self.baseMVA is None:
            self.baseMVA = params.get("baseMVA")
        if self.baseMVA_orig is None:
            self.baseMVA_orig = params.get("baseMVA_orig")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.baseMVA is None:
            raise ValueError("BaseMVA is not specified")

        if self.baseMVA == 0:
            raise ZeroDivisionError("BaseMVA is 0.")

        if self.node_data:
            data[:, PD] = data[:, PD] / self.baseMVA
            data[:, QD] = data[:, QD] / self.baseMVA
            data[:, PG] = data[:, PG] / self.baseMVA
            data[:, QG] = data[:, QG] / self.baseMVA
            data[:, VA] = data[:, VA] * torch.pi / 180.0
        else:
            data = data * self.baseMVA_orig / self.baseMVA

        return data

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        if self.baseMVA is None:
            raise ValueError("fit must be called before inverse_transform.")

        if self.node_data:
            normalized_data[:, PD] = normalized_data[:, PD] * self.baseMVA
            normalized_data[:, QD] = normalized_data[:, QD] * self.baseMVA
            normalized_data[:, PG] = normalized_data[:, PG] * self.baseMVA
            normalized_data[:, QG] = normalized_data[:, QG] * self.baseMVA
            normalized_data[:, VA] = normalized_data[:, VA] * 180.0 / torch.pi
        else:
            normalized_data = normalized_data * self.baseMVA / self.baseMVA_orig

        return normalized_data

    def get_stats(self) -> dict:
        return {
            "baseMVA": self.baseMVA,
            "baseMVA_orig": self.baseMVA_orig,
        }


@NORMALIZERS_REGISTRY.register("identity")
class IdentityNormalizer(Normalizer):
    """
    No normalization: returns data unchanged.

    Args:
            node_data: Whether data is node-level or edge-level
            args (NestedNamespace): Parameters
    """

    def __init__(self, node_data: bool, args):
        pass

    def fit(self, data: torch.Tensor) -> dict:
        return {}

    def fit_from_dict(self, params: dict):
        pass

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        return normalized_data

    def get_stats(self) -> dict:
        return {"note": "No normalization applied."}

@NORMALIZERS_REGISTRY.register("heterobaseMVAnorm")
class HeteroBaseMVANormalizer(Normalizer):
    """
    In power systems, a suitable normalization strategy must preserve the physical properties of
    the system. A known method is the conversion to the per-unit (p.u.) system, which expresses
    electrical quantities such as voltage, current, power, and impedance as fractions of predefined
    base values. These base values are usually chosen based on system parameters, such as rated
    voltage. The per-unit conversion ensures that power system equations remain scale-invariant,
    preserving fundamental physical relationships.
    """

    def __init__(self, node_data: bool, args):
        """
        Args:
            node_data: Whether data is node-level or edge-level
            args (NestedNamespace): Parameters

        Attributes:
            baseMVA (float): baseMVA found in casefile. From ``args.data.baseMVA``.
        """
        self.node_data = node_data
        self.baseMVA_orig = 100.0
        self.baseMVA = None

    def to(self, device):
        pass

    
    def fit(self, baseMVA: Optional[float] = None, bus_data: Optional[pd.DataFrame] = None, gen_data: Optional[pd.DataFrame] = None) -> dict:
        """
        Fit method for both node-level and edge-level normalization.

        - For node-level: pass `bus_data` and `gen_data`.
        - For edge-level: pass `baseMVA` directly.
        """
        if self.node_data:
            if bus_data is None or gen_data is None:
                raise ValueError("bus_data and gen_data must be provided for node-level normalization.")

            pd_values = bus_data["Pd"]
            qd_values = bus_data["Qd"]
            pg_values = gen_data["p_mw"]
            qg_values = bus_data["Qg"]

            non_zero_values = pd.concat([
                pd_values[pd_values != 0],
                qd_values[qd_values != 0],
                pg_values[pg_values != 0],
                qg_values[qg_values != 0]
            ])

            self.baseMVA = np.percentile(non_zero_values, 95)
        else:
            if baseMVA is None:
                raise ValueError("baseMVA must be provided for edge-level normalization.")
            self.baseMVA = baseMVA

        return {"baseMVA_orig": self.baseMVA_orig, "baseMVA": self.baseMVA}


    def fit_from_dict(self, params: dict):
        if self.baseMVA is None:
            self.baseMVA = params.get("baseMVA")
        if self.baseMVA_orig is None:
            self.baseMVA_orig = params.get("baseMVA_orig")

    def transform(self, bus_data: torch.Tensor = None, gen_data: torch.Tensor = None, edge_data: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.baseMVA is None:
            raise ValueError("BaseMVA is not specified")
        if self.baseMVA == 0:
            raise ZeroDivisionError("BaseMVA is 0.")

        if self.node_data:
            if bus_data is None or gen_data is None:
                raise ValueError("bus_data and gen_data must be provided for node-level normalization.")

            # Normalize bus data
            bus_data[:, PD_H] = bus_data[:, PD_H] / self.baseMVA
            bus_data[:, QD_H] = bus_data[:, QD_H] / self.baseMVA
            bus_data[:, QG_H] = bus_data[:, QG_H] / self.baseMVA
            bus_data[:, MIN_QG_H] = bus_data[:, MIN_QG_H] / self.baseMVA
            bus_data[:, MAX_QG_H] = bus_data[:, MAX_QG_H] / self.baseMVA
            bus_data[:, VA_H] = bus_data[:, VA_H] * torch.pi / 180.0

            # Normalize generator data
            gen_data[:, PG_H] = gen_data[:, PG_H] / self.baseMVA
            gen_data[:, MIN_PG] = gen_data[:, MIN_PG] / self.baseMVA
            gen_data[:, MAX_PG] = gen_data[:, MAX_PG] / self.baseMVA
            gen_data[:, C0_H] = torch.sign(gen_data[:, C0_H]) * torch.log1p(torch.abs(gen_data[:, C0_H]))
            gen_data[:, C1_H] = torch.sign(gen_data[:, C1_H]) * torch.log1p(torch.abs(gen_data[:, C1_H]))
            gen_data[:, C2_H] = torch.sign(gen_data[:, C2_H]) * torch.log1p(torch.abs(gen_data[:, C2_H]))
            

            return bus_data, gen_data

        else:
            if edge_data is None:
                raise ValueError("edge_data must be provided for edge-level normalization.")

            edge_data = edge_data * self.baseMVA_orig / self.baseMVA
            return edge_data

    def inverse_transform(self, bus_data: torch.Tensor = None, gen_data: torch.Tensor = None, edge_data: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.baseMVA is None:
            raise ValueError("fit must be called before inverse_transform.")

        if self.node_data:
            if bus_data is None or gen_data is None:
                raise ValueError("bus_data and gen_data must be provided for node-level inverse normalization.")

            bus_data[:, PD_H] = bus_data[:, PD_H] * self.baseMVA
            bus_data[:, QD_H] = bus_data[:, QD_H] * self.baseMVA
            bus_data[:, QG_H] = bus_data[:, QG_H] * self.baseMVA
            bus_data[:, MIN_QG_H] = bus_data[:, MIN_QG_H] * self.baseMVA
            bus_data[:, MAX_QG_H] = bus_data[:, MAX_QG_H] * self.baseMVA
            bus_data[:, VA_H] = bus_data[:, VA_H] * 180.0 / torch.pi

            gen_data[:, PG_H] = gen_data[:, PG_H] * self.baseMVA
            gen_data[:, MIN_PG] = gen_data[:, MIN_PG] * self.baseMVA
            gen_data[:, MAX_PG] = gen_data[:, MAX_PG] * self.baseMVA
            gen_data[:, C0_H] = torch.sign(gen_data[:, C0_H]) * (torch.exp(torch.abs(gen_data[:, C0_H])) - 1)
            gen_data[:, C1_H] = torch.sign(gen_data[:, C1_H]) * (torch.exp(torch.abs(gen_data[:, C1_H])) - 1)
            gen_data[:, C2_H] = torch.sign(gen_data[:, C2_H]) * (torch.exp(torch.abs(gen_data[:, C2_H])) - 1)

            return bus_data, gen_data

        else:
            if edge_data is None:
                raise ValueError("edge_data must be provided for edge-level inverse normalization.")

            edge_data = edge_data * self.baseMVA / self.baseMVA_orig
            return edge_data

    def get_stats(self) -> dict:
        return {
            "baseMVA": self.baseMVA,
            "baseMVA_orig": self.baseMVA_orig,
        }
@NORMALIZERS_REGISTRY.register("heterobaseMVAnormBranch")
class HeteroBaseMVANormalizerBranch(Normalizer):
    """
    In power systems, a suitable normalization strategy must preserve the physical properties of
    the system. A known method is the conversion to the per-unit (p.u.) system, which expresses
    electrical quantities such as voltage, current, power, and impedance as fractions of predefined
    base values. These base values are usually chosen based on system parameters, such as rated
    voltage. The per-unit conversion ensures that power system equations remain scale-invariant,
    preserving fundamental physical relationships.
    """

    def __init__(self, node_data: bool, args):
        """
        Args:
            node_data: Whether data is node-level or edge-level
            args (NestedNamespace): Parameters

        Attributes:
            baseMVA (float): baseMVA found in casefile. From ``args.data.baseMVA``.
        """
        self.node_data = node_data
        self.baseMVA_orig = getattr(args.data, "baseMVA", 100)
        self.baseMVA = None

    def to(self, device):
        pass

    
    def fit(self, baseMVA: Optional[float] = None, bus_data: Optional[pd.DataFrame] = None, gen_data: Optional[pd.DataFrame] = None) -> dict:
        """
        Fit method for both node-level and edge-level normalization.

        - For node-level: pass `bus_data` and `gen_data`.
        - For edge-level: pass `baseMVA` directly.
        """
        if self.node_data:
            if bus_data is None or gen_data is None:
                raise ValueError("bus_data and gen_data must be provided for node-level normalization.")

            pd_values = bus_data["Pd"]
            qd_values = bus_data["Qd"]
            pg_values = gen_data["p_mw"]
            qg_values = bus_data["Qg"]

            non_zero_values = pd.concat([
                pd_values[pd_values != 0],
                qd_values[qd_values != 0],
                pg_values[pg_values != 0],
                qg_values[qg_values != 0]
            ])

            self.baseMVA = np.percentile(non_zero_values, 95)
        else:
            if baseMVA is None:
                raise ValueError("baseMVA must be provided for edge-level normalization.")
            self.baseMVA = baseMVA

        return {"baseMVA_orig": self.baseMVA_orig, "baseMVA": self.baseMVA}


    def fit_from_dict(self, params: dict):
        if self.baseMVA is None:
            self.baseMVA = params.get("baseMVA")
        if self.baseMVA_orig is None:
            self.baseMVA_orig = params.get("baseMVA_orig")

    def transform(
        self,
        bus_data: torch.Tensor = None,
        gen_data: torch.Tensor = None,
        edge_data: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.baseMVA is None or self.baseMVA == 0:
            raise ValueError("BaseMVA not properly set")

        if self.node_data:
            if bus_data is None or gen_data is None:
                raise ValueError("bus_data and gen_data must be provided for node normalization")

            # --- Node normalization ---
            bus_data[:, PD_H] /= self.baseMVA
            bus_data[:, QD_H] /= self.baseMVA
            bus_data[:, QG_H] /= self.baseMVA
            bus_data[:, MIN_QG_H] /= self.baseMVA
            bus_data[:, MAX_QG_H] /= self.baseMVA
            bus_data[:, VA_H] *= torch.pi / 180.0
            bus_data[:, GS] *= self.baseMVA_orig / self.baseMVA
            bus_data[:, BS] *= self.baseMVA_orig / self.baseMVA

            gen_data[:, PG_H] /= self.baseMVA
            gen_data[:, MIN_PG] /= self.baseMVA
            gen_data[:, MAX_PG] /= self.baseMVA
            gen_data[:, C0_H] = torch.sign(gen_data[:, C0_H]) * torch.log1p(torch.abs(gen_data[:, C0_H]))
            gen_data[:, C1_H] = torch.sign(gen_data[:, C1_H]) * torch.log1p(torch.abs(gen_data[:, C1_H]))
            gen_data[:, C2_H] = torch.sign(gen_data[:, C2_H]) * torch.log1p(torch.abs(gen_data[:, C2_H]))

            return bus_data, gen_data

        else:
            if edge_data is None:
                raise ValueError("edge_data must be provided for edge normalization")

            # --- Edge normalization ---
            # Power flows
            edge_data[:, P_E] /= self.baseMVA
            edge_data[:, Q_E] /= self.baseMVA

            # Admittances
            edge_data[:, YFF_TT_R:YFT_TF_I + 1] *= self.baseMVA_orig / self.baseMVA

            # Tap ratio unchanged
            # Angles to radians
            edge_data[:, ANG_MIN] *= torch.pi / 180.0
            edge_data[:, ANG_MAX] *= torch.pi / 180.0

            # Thermal limit
            edge_data[:, RATE_A] /= self.baseMVA

            # BR_STATUS unchanged
            return edge_data

    def inverse_transform(
        self,
        bus_data: torch.Tensor = None,
        gen_data: torch.Tensor = None,
        edge_data: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.node_data:
            if bus_data is None or gen_data is None:
                raise ValueError("bus_data and gen_data must be provided for inverse node normalization")

            bus_data[:, PD_H] *= self.baseMVA
            bus_data[:, QD_H] *= self.baseMVA
            bus_data[:, QG_H] *= self.baseMVA
            bus_data[:, MIN_QG_H] *= self.baseMVA
            bus_data[:, MAX_QG_H] *= self.baseMVA
            bus_data[:, VA_H] *= 180.0 / torch.pi
            bus_data[:, GS] *= self.baseMVA / self.baseMVA_orig
            bus_data[:, BS] *= self.baseMVA / self.baseMVA_orig

            gen_data[:, PG_H] *= self.baseMVA
            gen_data[:, MIN_PG] *= self.baseMVA
            gen_data[:, MAX_PG] *= self.baseMVA
            gen_data[:, C0_H] = torch.sign(gen_data[:, C0_H]) * (torch.exp(torch.abs(gen_data[:, C0_H])) - 1)
            gen_data[:, C1_H] = torch.sign(gen_data[:, C1_H]) * (torch.exp(torch.abs(gen_data[:, C1_H])) - 1)
            gen_data[:, C2_H] = torch.sign(gen_data[:, C2_H]) * (torch.exp(torch.abs(gen_data[:, C2_H])) - 1)

            return bus_data, gen_data

        else:
            if edge_data is None:
                raise ValueError("edge_data must be provided for inverse edge normalization")

            edge_data[:, P_E] *= self.baseMVA
            edge_data[:, Q_E] *= self.baseMVA

            edge_data[:, YFF_TT_R:YFT_TF_I + 1] *= self.baseMVA / self.baseMVA_orig

            edge_data[:, ANG_MIN] *= 180.0 / torch.pi
            edge_data[:, ANG_MAX] *= 180.0 / torch.pi
            edge_data[:, RATE_A] *= self.baseMVA

            return edge_data

    def get_stats(self) -> dict:
        return {
            "baseMVA": self.baseMVA,
            "baseMVA_orig": self.baseMVA_orig,
        }
