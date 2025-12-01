import torch
from torch import nn
from gridfm_graphkit.datasets.globals import *
from torch_scatter import scatter_add

class ComputeBranchFlow(nn.Module):
    """Compute sending-end branch flows (Pf, Qf) for all branches."""
    def forward(self, bus_data, edge_index, edge_attr):
        from_idx, to_idx = edge_index

        # Voltage magnitudes and angles
        Vf_mag, Vf_ang = bus_data[from_idx, VM_H], bus_data[from_idx, VA_H]
        Vt_mag, Vt_ang = bus_data[to_idx, VM_H], bus_data[to_idx, VA_H]

        # Real & imaginary voltage components
        Vf_r = Vf_mag * torch.cos(Vf_ang)
        Vf_i = Vf_mag * torch.sin(Vf_ang)
        Vt_r = Vt_mag * torch.cos(Vt_ang)
        Vt_i = Vt_mag * torch.sin(Vt_ang)

        # Branch admittance components
        Yfftt_r, Yfftt_i = edge_attr[:, YFF_TT_R], edge_attr[:, YFF_TT_I]
        Yfttf_r, Yfttf_i = edge_attr[:, YFT_TF_R], edge_attr[:, YFT_TF_I]

        # Sending-end currents
        Ift_r = Yfftt_r * Vf_r - Yfftt_i * Vf_i + Yfttf_r * Vt_r - Yfttf_i * Vt_i
        Ift_i = Yfftt_r * Vf_i + Yfftt_i * Vf_r + Yfttf_r * Vt_i + Yfttf_i * Vt_r

        # Sending-end power flows
        Pft = Vf_r * Ift_r + Vf_i * Ift_i
        Qft = Vf_i * Ift_r - Vf_r * Ift_i

        return Pft, Qft



class ComputeNodeInjection(nn.Module):
    """Aggregate branch flows into node-level incoming injections."""
    def forward(self, Pft, Qft, edge_index, num_bus):
        """
        Args:
            Pft, Qft: [num_edges] branch flows
            edge_index: [2, num_edges] (from_bus, to_bus)
            num_bus: number of bus nodes
        Returns:
            P_in, Q_in: aggregated incoming power per bus
        """
        from_idx, _ = edge_index  # Only sending end contributes as "incoming" to node
        P_in = scatter_add(Pft, from_idx, dim=0, dim_size=num_bus)
        Q_in = scatter_add(Qft, from_idx, dim=0, dim_size=num_bus)

        return P_in, Q_in

    
class ComputeNodeResiduals(nn.Module):
    """Compute net residuals per bus combining branch flows, generators, loads, and shunts."""
    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig, agg_bus):
        # Shunt contributions
        p_shunt = - bus_data_orig[:, GS] * bus_data_pred[:, VM_H]**2
        q_shunt = bus_data_orig[:, BS] * bus_data_pred[:, VM_H]**2

        # Net residuals per bus
        residual_P = agg_bus - bus_data_pred[:, PD_H] + p_shunt - P_in
        residual_Q = bus_data_pred[:, QG_H] - bus_data_pred[:, QD_H] + q_shunt - Q_in

        return residual_P, residual_Q



def bound_with_sigmoid(pred, low, high):
    return low + (high - low) * torch.sigmoid(pred)