import torch
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv
from gridfm_graphkit.training.utils import PowerFlowResidualLayer
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.datasets.globals import *
from torch_geometric.nn import MessagePassing

class GenToBusAggregator(MessagePassing):
    """
    A parameter-free message passing layer that aggregates generator outputs
    (e.g., P_G, Q_G) to their connected buses
    """
    def __init__(self, aggr="add"):
        super().__init__(aggr=aggr)

    def forward(self, x_gen, edge_index_gen2bus, num_bus):
        """
        Args:
            x_gen: [num_gen, F] generator features (e.g., Pg, Qg)
            edge_index_gen2bus: [2, num_edges] where src=gen, dst=bus
            num_bus: total number of bus nodes (for dimension of result)
        Returns:
            agg_bus: [num_bus, F] sum of generator info per bus
        """
        return self.propagate(edge_index_gen2bus, x=x_gen, size=(x_gen.size(0), num_bus))

    def message(self, x_j):
        return x_j

def bound_with_sigmoid(pred, low, high):
    return low + (high - low) * torch.sigmoid(pred)


@MODELS_REGISTRY.register("GNN_PBE_HeteroTransformerConv")
class GNN_PBE_HeteroTransformerConv(nn.Module):
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
        self.engine = PowerFlowResidualLayer()
        self.gen2bus_agg = GenToBusAggregator()

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
        gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]
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


        agg_gen_on_bus = self.gen2bus_agg(x_dict["gen"][:,:(PG_H+1)], gen_to_bus_index, num_bus)
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


            # Aggregate gen predictions to buses
            agg_gen_on_bus = self.gen2bus_agg(gen_temp, gen_to_bus_index, num_bus)

            temp_output = torch.cat([bus_temp, agg_gen_on_bus], dim=1)

            # Compute residuals
            complex_res_bus = self.engine(
                temp_output,
                bus_edge_index,
                bus_edge_attr,
            )
            residual_real_bus = torch.real(complex_res_bus)   
            residual_imag_bus = torch.imag(complex_res_bus)
            bus_residuals = torch.stack([residual_real_bus, residual_imag_bus], dim=-1)

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

        agg_gen_on_bus = self.gen2bus_agg(final_gen_out, gen_to_bus_index, num_bus)
        temp_output = torch.cat([final_bus_out, agg_gen_on_bus], dim=1)
        # Compute residuals
        final_complex_res_bus = self.engine(
            temp_output,
            bus_edge_index,
            bus_edge_attr,
        )
        final_residual_real_bus = torch.real(final_complex_res_bus)   
        final_residual_imag_bus = torch.imag(final_complex_res_bus)
        final_bus_residuals = torch.stack([final_residual_real_bus, final_residual_imag_bus], dim=-1)

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
        

