from copy import deepcopy
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
from gridfm_graphkit.datasets.globals import (
    # Edge indices
    G,
    B,
    # Generator feature indices
    G_ON,
    # Edge feature indices
    B_ON,
    YFF_TT_I,
    YFF_TT_R,
    YFT_TF_I,
    YFT_TF_R,
)
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer


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


class RemoveInactiveGenerators(BaseTransform):
    """
    Removes generators where G_ON == 0.
    Uses the global index G_ON to access generator on/off flag.
    """

    def forward(self, data):
        # Mask of generators that are ON
        active_mask = data["gen"].x[:, G_ON] == 1

        num_gen = data["gen"].num_nodes

        # Mapping old generator IDs → new compact IDs
        old_to_new = torch.full((num_gen,), -1, dtype=torch.long)
        old_to_new[active_mask] = torch.arange(active_mask.sum())

        # Filter generator node features
        data["gen"].x = data["gen"].x[active_mask]
        data["gen"].x = data["gen"].x[:, :G_ON]
        data["gen"].y = data["gen"].y[active_mask]

        # ---- Update hetero edges ----

        # gen → bus edges
        e = data["gen", "connected_to", "bus"].edge_index
        keep = active_mask[e[0]]  # generator is source
        new_e = e[:, keep].clone()
        new_e[0] = old_to_new[new_e[0]]
        data["gen", "connected_to", "bus"].edge_index = new_e

        # bus → gen edges
        e = data["bus", "connected_to", "gen"].edge_index
        keep = active_mask[e[1]]  # generator is target
        new_e = e[:, keep].clone()
        new_e[1] = old_to_new[new_e[1]]
        data["bus", "connected_to", "gen"].edge_index = new_e

        return data


class RemoveInactiveBranches(BaseTransform):
    """
    Removes branches where B_ON == 0.
    Uses global index B_ON in edge_attr.
    """

    def forward(self, data):
        et = ("bus", "connects", "bus")

        # Mask for active (in-service) branches
        active_mask = data[et].edge_attr[:, B_ON] == 1

        # Apply the mask
        data[et].edge_index = data[et].edge_index[:, active_mask]
        data[et].edge_attr = data[et].edge_attr[active_mask]
        data[et].edge_attr = data[et].edge_attr[:, :B_ON]
        data[et].y = data[et].y[active_mask]

        return data


class ApplyMasking(BaseTransform):
    """
    Apply masking to data
    """

    def __init__(self, args):
        super().__init__()
        self.mask_value = args.data.mask_value

    def forward(self, data):
        data.x_dict["bus"][data.mask_dict["bus"]] = self.mask_value
        data.x_dict["gen"][data.mask_dict["gen"]] = self.mask_value
        data.edge_attr_dict[("bus", "connects", "bus")][data.mask_dict["branch"]] = (
            self.mask_value
        )

        return data


class LoadGridParamsFromPath(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.grid_path = args.task.grid_path
        self.grid_data = torch.load(self.grid_path, weights_only=False)

        # Normalizer is needed in order to normalize the grid_data in case the input data is normalized
        self.normalizer = HeteroDataMVANormalizer(args)

        # Set to a dummy value since it is needed for the normalizer transform, but the column vn_kv will be ignored.
        self.normalizer.vn_kv_max = 1

    def forward(self, data):
        if hasattr(data, "is_normalized"):
            self.normalizer.baseMVA = data.baseMVA
            grid_data = deepcopy(self.grid_data)
            self.normalizer.transform(grid_data)

        cols = [YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I, B_ON]
        data[("bus", "connects", "bus")].edge_attr[:, cols] = grid_data[
            ("bus", "connects", "bus")
        ].edge_attr[:, cols]
        data["gen"].x[:, G_ON] = grid_data["gen"].x[:, G_ON]
        return data
