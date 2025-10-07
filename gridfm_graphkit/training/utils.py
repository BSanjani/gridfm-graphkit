from gridfm_graphkit.datasets.globals import PD, QD, PG, QG, VM, VA, G, B
import torch
from torch_geometric.utils import to_torch_coo_tensor


def compute_node_residuals(x, edge_index, edge_attr, mask=None, target=None):
    """
    Computes node-wise active/reactive power flow residuals.
    """

    # Clone predictions to avoid overwriting the original tensor.
    # This is important when combining multiple losses during training (e.g., MSE loss + Power Balance Equation Loss),
    # since modifying the tensor in-place would affect other loss components that rely on the original predictions.
    if mask is not None and target is not None:
        temp_x = x.clone()

        # If a value is not masked, then use the original one
        temp_x[~mask] = target[~mask]
        x = temp_x

    Vm, Va = x[:, VM], x[:, VA]
    Pd, Qd = x[:, PD], x[:, QD]
    Pg, Qg = x[:, PG], x[:, QG]

    V = Vm * torch.exp(1j * Va)
    V_conj = torch.conj(V)

    edge_complex = edge_attr[:, G] + 1j * edge_attr[:, B]
    Y_bus_sparse = to_torch_coo_tensor(
        edge_index,
        edge_complex,
        size=(x.size(0), x.size(0)),
    )
    Y_bus_conj = torch.conj(Y_bus_sparse)

    indices = torch.arange(V.size(0), device=V.device)
    indices = torch.stack([indices, indices])
    V_diag_sparse = torch.sparse_coo_tensor(
        indices,
        V,
        size=(V.size(0), V.size(0)),
        dtype=torch.complex64,
    )

    S_injection = V_diag_sparse @ Y_bus_conj @ V_conj
    S_net_power_balance = (Pg - Pd) + 1j * (Qg - Qd)

    residual_complex = S_net_power_balance - S_injection
    return residual_complex
