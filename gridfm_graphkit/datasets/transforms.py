from gridfm_graphkit.datasets.globals import *
from gridfm_graphkit.io.registries import MASKING_REGISTRY

import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform
from typing import Optional
import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.utils import (
    get_self_loop_attr,
    is_torch_sparse_tensor,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


class AddNormalizedRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding from the
    [Graph Neural Networks with Learnable Structural and Positional Representations](https://arxiv.org/abs/2110.07875)
    paper to the given graph. This is an adaptation from the original Pytorch Geometric implementation.

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """

    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = "random_walk_pe",
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        if data.edge_index is None:
            raise ValueError("Expected data.edge_index to be not None")
        row, col = data.edge_index
        N = data.num_nodes
        if N is None:
            raise ValueError("Expected data.num_nodes to be not None")

        if N <= 2_000:  # Dense code path for faster computation:
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = data.edge_weight
            loop_index = torch.arange(N, device=row.device)
        elif torch_geometric.typing.WITH_WINDOWS:
            adj = to_torch_coo_tensor(
                data.edge_index,
                data.edge_weight,
                size=data.size(),
            )
        else:
            adj = to_torch_csr_tensor(
                data.edge_index,
                data.edge_weight,
                size=data.size(),
            )

        row_sums = adj.sum(dim=1, keepdim=True)  # Sum along rows
        row_sums = row_sums.clamp(min=1e-8)  # Prevent division by zero

        adj = adj / row_sums  # Normalize each row to sum to 1

        def get_pe(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
            return out[loop_index, loop_index]

        out = adj
        pe_list = [get_pe(out)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_pe(out))

        pe = torch.stack(pe_list, dim=-1)
        data[self.attr_name] = pe

        return data


class AddEdgeWeights(BaseTransform):
    """
    Computes and adds edge weight as the magnitude of complex admittance.

    The magnitude is computed from the G and B components in `data.edge_attr` and stored in `data.edge_weight`.
    """

    def forward(self, data):
        if not hasattr(data, "edge_attr"):
            raise AttributeError("Data must have 'edge_attr'.")

        # Extract real and imaginary parts of admittance
        real = data.edge_attr[:, G]
        imag = data.edge_attr[:, B]

        # Compute the magnitude of the complex admittance
        edge_weight = torch.sqrt(real**2 + imag**2)

        # Add the computed edge weights to the data object
        data.edge_weight = edge_weight

        return data


@MASKING_REGISTRY.register("none")
class AddIdentityMask(BaseTransform):
    """Creates an identity mask, and adds it as a `mask` attribute.

    The mask is generated such that every entry is False, so no masking is actually applied
    """

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        # Generate an identity mask
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        # Add the mask to the data object
        data.mask = mask

        return data


@MASKING_REGISTRY.register("rnd")
class AddRandomMask(BaseTransform):
    """Creates a random mask, and adds it as a `mask` attribute.

    The mask is generated such that each entry is `True` with probability
    `mask_ratio` and `False` otherwise.
    """

    def __init__(self, args):
        super().__init__()
        self.mask_dim = args.data.mask_dim
        self.mask_ratio = args.data.mask_ratio

    def forward(self, data):
        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate a random mask
        mask = torch.rand(data.x.size(0), self.mask_dim) < self.mask_ratio

        # Add the mask to the data object
        data.mask = mask

        return data

@MASKING_REGISTRY.register("pf_hetero")
class AddPFHeteroMask(BaseTransform):
    """Creates masks for a heterogeneous power flow graph."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        bus_x = data.x_dict["bus"]
        gen_x = data.x_dict["gen"]
        num_buses = bus_x.size(0)

        # Identify bus types
        mask_PQ = bus_x[:, PQ_H] == 1
        mask_PV = bus_x[:, PV_H] == 1
        mask_REF = bus_x[:, REF_H] == 1

        # Initialize mask tensors
        mask_bus = torch.zeros_like(bus_x, dtype=torch.bool)
        mask_gen = torch.zeros_like(gen_x, dtype=torch.bool)
        mask_out = torch.zeros((num_buses, 6), dtype=torch.bool)

        mask_bus[:, MIN_VM_H] = True
        mask_bus[:, MAX_VM_H] = True
        mask_bus[:, MIN_QG_H] = True
        mask_bus[:, MAX_QG_H] = True

        mask_gen[:, MIN_PG] = True
        mask_gen[:, MAX_PG] = True
        mask_gen[:, C0_H] = True
        mask_gen[:, C1_H] = True
        mask_gen[:, C2_H] = True


        # --- PQ buses ---
        mask_bus[mask_PQ, VM_H] = True
        mask_bus[mask_PQ, VA_H] = True
        mask_out[mask_PQ, VM_H] = True
        mask_out[mask_PQ, VA_H] = True

        # --- PV buses ---
        mask_bus[mask_PV, VA_H] = True
        mask_bus[mask_PV, QG_H] = True
        mask_out[mask_PV, VA_H] = True
        mask_out[mask_PV, QG_H] = True

        # --- REF buses ---
        mask_bus[mask_REF, QG_H] = True
        mask_out[mask_REF, 5] = True  # PG_H for REF
        mask_out[mask_REF, QG_H] = True

        # --- Generators connected to REF buses ---
        gen_bus_edges = data.edge_index_dict[("gen", "connected_to", "bus")]
        gen_indices, bus_indices = gen_bus_edges
        ref_gens = gen_indices[mask_REF[bus_indices]]
        mask_gen[ref_gens, PG_H] = True

        data.mask_dict = {
            "bus": mask_bus,
            "gen": mask_gen,
            "out": mask_out
        }

        return data

@MASKING_REGISTRY.register("opf_hetero")
class AddOPFHeteroMask(BaseTransform):
    """Creates masks for a heterogeneous power flow graph."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        bus_x = data.x_dict["bus"]
        gen_x = data.x_dict["gen"]

        # Identify bus types
        mask_PQ = bus_x[:, PQ_H] == 1
        mask_PV = bus_x[:, PV_H] == 1
        mask_REF = bus_x[:, REF_H] == 1

        # Initialize mask tensors
        mask_bus = torch.zeros_like(bus_x, dtype=torch.bool)
        mask_gen = torch.zeros_like(gen_x, dtype=torch.bool)


        # --- PQ buses ---
        mask_bus[mask_PQ, VM_H] = True
        mask_bus[mask_PQ, VA_H] = True

        # --- PV buses ---
        mask_bus[mask_PV, VA_H] = True
        mask_bus[mask_PV, VM_H] = True
        mask_bus[mask_PV, QG_H] = True

        # --- REF buses ---
        mask_bus[mask_REF, QG_H] = True
        mask_bus[mask_REF, VM_H] = True
        mask_bus[mask_REF, VA_H] = True

        mask_gen[:, PG_H] = True

        data.mask_dict = {
            "bus": mask_bus,
            "gen": mask_gen,
        }

        return data

@MASKING_REGISTRY.register("pf")
class AddPFMask(BaseTransform):
    """Creates a mask according to the power flow problem and assigns it as a `mask` attribute."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate masks for each type of node
        mask_PQ = data.x[:, PQ] == 1  # PQ buses
        mask_PV = data.x[:, PV] == 1  # PV buses
        mask_REF = data.x[:, REF] == 1  # Reference buses

        # Initialize the mask tensor with False values
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        mask[mask_PQ, VM] = True  # Mask Vm for PQ buses
        mask[mask_PQ, VA] = True  # Mask Va for PQ buses

        mask[mask_PV, QG] = True  # Mask Qg for PV buses
        mask[mask_PV, VA] = True  # Mask Va for PV buses

        mask[mask_REF, PG] = True  # Mask Pg for REF buses
        mask[mask_REF, QG] = True  # Mask Qg for REF buses

        # Attach the mask to the data object
        data.mask = mask

        return data


@MASKING_REGISTRY.register("opf")
class AddOPFMask(BaseTransform):
    """Creates a mask according to the optimal power flow problem and assigns it as a `mask` attribute."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate masks for each type of node
        mask_PQ = data.x[:, PQ] == 1  # PQ buses
        mask_PV = data.x[:, PV] == 1  # PV buses
        mask_REF = data.x[:, REF] == 1  # Reference buses

        # Initialize the mask tensor with False values
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        mask[mask_PQ, VM] = True  # Mask Vm for PQ
        mask[mask_PQ, VA] = True  # Mask Va for PQ

        mask[mask_PV, PG] = True  # Mask Pg for PV
        mask[mask_PV, QG] = True  # Mask Qg for PV
        mask[mask_PV, VM] = True  # Mask Vm for PV
        mask[mask_PV, VA] = True  # Mask Va for PV

        mask[mask_REF, PG] = True  # Mask Pg for REF
        mask[mask_REF, QG] = True  # Mask Qg for REF
        mask[mask_REF, VM] = True  # Mask Vm for REF
        mask[mask_REF, VA] = True  # Mask Va for REF

        # Attach the mask to the data object
        data.mask = mask

        return data
