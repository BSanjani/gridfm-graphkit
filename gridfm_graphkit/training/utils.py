from gridfm_graphkit.datasets.globals import VM, VA, PD, QD, PG, QG, B, G
import torch
from torch_geometric.nn import MessagePassing


class PowerFlowResidualLayerHomo(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index, edge_attr, mask=None, target=None):
        if mask is not None and target is not None:
            temp_x = x.clone()

            # If a value is not masked, then use the original one
            temp_x[~mask] = target[~mask]
            x = temp_x

        Vm, Va = x[:, VM], x[:, VA]
        Pd, Qd = x[:, PD], x[:, QD]
        Pg, Qg = x[:, PG], x[:, QG]

        V = Vm * torch.exp(1j * Va)
        V = V.unsqueeze(-1)

        # Compute messages and aggregate
        S_injection = self.propagate(edge_index, x=V, edge_attr=edge_attr).squeeze()

        # Compute net power balance
        S_net_power_balance = (Pg - Pd) + 1j * (Qg - Qd)

        residual_complex = S_net_power_balance - S_injection
        return residual_complex

    def message(self, x_i, x_j, edge_attr):
        Y_ij = edge_attr[:, G] - 1j * edge_attr[:, B]
        result = x_i.squeeze() * Y_ij * torch.conj(x_j.squeeze())
        return result.unsqueeze(-1)
