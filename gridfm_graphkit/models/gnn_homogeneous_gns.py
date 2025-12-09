import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from gridfm_graphkit.training.utils import PowerFlowResidualLayerHomo
from gridfm_graphkit.io.registries import MODELS_REGISTRY


@MODELS_REGISTRY.register("GNS_homogeneous")
class GNS_homogeneous(nn.Module):
    """Graph neural network with enhanced physics integration."""

    def __init__(self, args) -> None:
        super().__init__()
        self.num_layers = args.model.num_layers
        self.hidden_dim = args.model.hidden_size
        self.input_dim = args.model.input_dim
        self.output_dim = args.model.output_dim
        self.edge_dim = args.model.edge_dim
        self.heads = getattr(args.model, "attention_head", 1)
        self.mask_dim = getattr(args.data, "mask_dim", 6)
        self.mask_value = getattr(args.data, "mask_value", -1.0)
        self.learn_mask = getattr(args.data, "learn_mask", True)
        self.dropout = getattr(args.model, "dropout", 0.0)
        self.last_residual = None

        self.layer_residuals = {}
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_edge = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.physics_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_dim * self.heads),
            nn.LeakyReLU(),
        )

        self.layers = nn.ModuleList(
            [
                TransformerConv(
                    self.hidden_dim if i == 0 else self.hidden_dim * self.heads,
                    self.hidden_dim,
                    heads=self.heads,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    beta=True,
                )
                for i in range(self.num_layers)
            ],
        )

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.hidden_dim * self.heads)
                for _ in range(self.num_layers)
            ],
        )

        self.mlp_shared = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.mask_value = nn.Parameter(
            torch.zeros(self.mask_dim) + self.mask_value,
            requires_grad=False,
        )

        self.activation = nn.LeakyReLU()

        self.engine = PowerFlowResidualLayerHomo()

    def forward(
        self,
        x: torch.Tensor,
        pe: torch.Tensor,  # not used
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        self.layer_residuals = {}

        # initial feature projection
        h = self.input_proj(x)
        edge_attr_encoded = self.input_proj_edge(edge_attr)

        for i, conv in enumerate(self.layers):
            h_new = self.activation(
                self.norms[i](conv(h, edge_index, edge_attr_encoded)),
            )

            h = h + h_new if h_new.shape == h.shape else h_new

            temp_output = self.mlp_shared(h)

            complex_residual = self.engine(
                temp_output,
                edge_index,
                edge_attr,
                mask,
                x[:, :6],
            )
            residual_real = torch.real(complex_residual)
            residual_imag = torch.imag(complex_residual)
            residuals = torch.stack([residual_real, residual_imag], dim=-1)

            self.layer_residuals[i] = torch.linalg.norm(residuals, dim=-1).mean()
            h = h + self.physics_mlp(residuals)

        output = self.mlp_shared(h)

        final_complex_residual = self.engine(
            output,
            edge_index,
            edge_attr,
            mask,
            x[:, :6],
        )
        final_residual_real = torch.real(final_complex_residual)
        final_residual_imag = torch.imag(final_complex_residual)
        final_residuals = torch.stack(
            [final_residual_real, final_residual_imag],
            dim=-1,
        )

        self.layer_residuals[self.num_layers] = torch.linalg.norm(
            final_residuals,
            dim=-1,
        ).mean()

        return output
