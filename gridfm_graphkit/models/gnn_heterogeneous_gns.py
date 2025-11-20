import torch
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv
from gridfm_graphkit.io.registries import MODELS_REGISTRY
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


@MODELS_REGISTRY.register("GNS_heterogeneous")
class GNS_heterogeneous(nn.Module):
    """
    Heterogeneous version of your Transformer-based GNN for buses and generators.
    - Expects node features as dict: x_dict = {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
    - Expects edge_index_dict and edge_attr_dict with keys:
        ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
      (edge_attr only needed for bus-bus currently; other relations can be None)
    - Keeps the physics residual idea but splits it into bus-step and gen-step residuals.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.num_layers = args.model.num_layers
        self.hidden_dim = args.model.hidden_size
        self.input_bus_dim = args.model.input_bus_dim  
        self.input_gen_dim = args.model.input_gen_dim   
        self.output_bus_dim = args.model.output_bus_dim
        self.output_gen_dim = args.model.output_gen_dim
        self.edge_dim = args.model.edge_dim
        self.heads = args.model.attention_head
        self.task = args.data.mask_type
        self.dropout = getattr(args.model, "dropout", 0.0)

        # projections for each node type
        self.input_proj_bus = nn.Sequential(
            nn.Linear(self.input_bus_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_gen = nn.Sequential(
            nn.Linear(self.input_gen_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        #######################
        ### NEW ADDITION
        #######################
        self.input_proj_edge = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # a small physics MLP that will take residuals (real, imag) and return a correction
        self.physics_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_dim * self.heads),
            nn.LeakyReLU(),
        )

        # Build hetero layers: HeteroConv of TransformerConv per relation
        self.layers = nn.ModuleList()
        self.norms_bus = nn.ModuleList()
        self.norms_gen = nn.ModuleList()
        for i in range(self.num_layers):
            # in-channels depend on whether it is first layer (hidden_dim) or subsequent (hidden_dim * heads)
            in_bus = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            in_gen = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            out_dim = self.hidden_dim  # TransformerConv will output hidden_dim (per head reduction in HeteroConv call)

            # relation -> conv module mapping
            conv_dict = {
                ("bus", "connects", "bus"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("gen", "connected_to", "bus"): TransformerConv(
                    in_gen,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("bus", "connected_to", "gen"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
            }

            hetero_conv = HeteroConv(conv_dict, aggr="sum")
            self.layers.append(hetero_conv)

            # Norms for node representations (note: after HeteroConv each node type will have size out_dim * heads)
            self.norms_bus.append(nn.LayerNorm(out_dim * self.heads))
            self.norms_gen.append(nn.LayerNorm(out_dim * self.heads))

        # Separate shared MLPs to produce final bus/gen outputs (predictions y)
        self.mlp_bus = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_bus_dim),
        )

        self.mlp_gen = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_gen_dim),
        )

        # mask param (kept similar to your original)
        self.activation = nn.LeakyReLU()
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()
        self.node_residuals_layer = ComputeNodeResiduals()

        # container for monitoring residual norms per layer and type
        self.layer_residuals = {}

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        """
        x_dict: {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
        edge_index_dict: keys like ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
        edge_attr_dict: same keys -> edge attributes (bus-bus requires G,B)
        batch_dict: dict mapping node types to batch tensors (if using batching). Not used heavily here but kept for API parity.
        mask: optional mask per node (applies when computing residuals)
        """

        self.layer_residuals = {}

        # 1) initial projections
        h_bus = self.input_proj_bus(x_dict["bus"])   # [num_bus, hidden_dim]
        h_gen = self.input_proj_gen(x_dict["gen"])   # [num_gen, hidden_dim]

        num_bus = x_dict["bus"].size(0)
        _ , gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        #######################
        ### NEW ADDITION
        #######################
        edge_attr_proj_dict = {}
        for key, edge_attr in edge_attr_dict.items():
            if edge_attr is not None:
                edge_attr_proj_dict[key] = self.input_proj_edge(edge_attr)
            else:
                edge_attr_proj_dict[key] = None


        #agg_gen_on_bus = scatter_add(x_dict["gen"][:,PG_H], gen_to_bus_index, dim=0, dim_size=num_bus)
        #agg_target = torch.cat([x_dict["bus"][:,:5], agg_gen_on_bus], dim=1)

        bus_mask = mask_dict["bus"][:, :(VA_H+1)]
        gen_mask = mask_dict["gen"][:, :(PG_H+1)]
        bus_fixed = x_dict["bus"][:, :(VA_H+1)]
        gen_fixed = x_dict["gen"][:, :(PG_H+1)]

        

        # iterate layers
        for i, conv in enumerate(self.layers):
            out_dict = conv({"bus": h_bus, "gen": h_gen}, edge_index_dict, edge_attr_proj_dict)
            out_bus = out_dict["bus"]   # [Nb, hidden_dim * heads]
            out_gen = out_dict["gen"]   # [Ng, hidden_dim * heads]

            out_bus = self.activation(self.norms_bus[i](out_bus))
            out_gen = self.activation(self.norms_gen[i](out_gen))

            # skip connection
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            # Decode bus and generator predictions
            bus_temp = self.mlp_bus(h_bus)   # [Nb, 5]  -> Pd, Qd, Qg, Vm, Va
            gen_temp = self.mlp_gen(h_gen)   # [Ng, 1]  -> Pg
            bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
            gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)

            if self.task == 'opf_hetero':
                bus_temp[:, VM_H] = bound_with_sigmoid(bus_temp[:, VM_H], x_dict["bus"][:, MIN_VM_H], x_dict["bus"][:, MAX_VM_H])
                bus_temp[:, QG_H] = bound_with_sigmoid(bus_temp[:, QG_H], x_dict["bus"][:, MIN_QG_H], x_dict["bus"][:, MAX_QG_H])
                gen_temp[:, PG_H] = bound_with_sigmoid(gen_temp[:, PG_H], x_dict["gen"][:, MIN_PG], x_dict["gen"][:, MAX_PG])            

            Pft, Qft = self.branch_flow_layer(bus_temp, bus_edge_index, bus_edge_attr)
            P_in, Q_in = self.node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
            agg_bus = scatter_add(gen_temp.squeeze(), gen_to_bus_index, dim=0, dim_size=num_bus)
            residual_P, residual_Q = self.node_residuals_layer(P_in, Q_in, bus_temp, x_dict["bus"], agg_bus)

            bus_residuals = torch.stack([residual_P, residual_Q], dim=-1)

            # Save and project residuals to latent space  
            self.layer_residuals[i] = torch.linalg.norm(bus_residuals, dim=-1).mean()
            h_bus = h_bus + self.physics_mlp(bus_residuals) 

        # final outputs
        final_bus_out = self.mlp_bus(h_bus)
        final_gen_out = self.mlp_gen(h_gen)
        final_bus_out = torch.where(bus_mask, final_bus_out, bus_fixed)
        final_gen_out = torch.where(gen_mask, final_gen_out, gen_fixed)
        
        if self.task == 'opf_hetero':
            final_bus_out[:, VM_H] = bound_with_sigmoid(final_bus_out[:, VM_H], x_dict["bus"][:, MIN_VM_H], x_dict["bus"][:, MAX_VM_H])
            final_bus_out[:, QG_H] = bound_with_sigmoid(final_bus_out[:, QG_H], x_dict["bus"][:, MIN_QG_H], x_dict["bus"][:, MAX_QG_H])
            final_gen_out[:, PG_H] = bound_with_sigmoid(final_gen_out[:, PG_H], x_dict["gen"][:, MIN_PG], x_dict["gen"][:, MAX_PG])

        Pft, Qft = self.branch_flow_layer(final_bus_out, bus_edge_index, bus_edge_attr)
        P_in, Q_in = self.node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
        agg_bus = scatter_add(final_gen_out, gen_to_bus_index, dim=0, dim_size=num_bus)
        residual_P, residual_Q = self.node_residuals_layer(P_in, Q_in, final_bus_out, x_dict["bus"], agg_bus.squeeze())
        temp_output = torch.cat([final_bus_out, agg_bus], dim=1)

        final_bus_residuals = torch.stack([residual_P, residual_Q], dim=-1)

        # Save and project residuals to latent space  
        self.layer_residuals[self.num_layers] = torch.linalg.norm(
            final_bus_residuals,
            dim=-1,
        ).mean()
        if self.task == 'pf_hetero':
            return temp_output
        elif self.task == 'opf_hetero':
            return {
                "bus": final_bus_out,
                "gen": final_gen_out
            }
        

