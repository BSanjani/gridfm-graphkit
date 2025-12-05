import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import os
import pandas as pd

from lightning.pytorch.loggers import MLFlowLogger
from gridfm_graphkit.io.param_handler import load_model, get_loss_function
import torch.nn.functional as F
from gridfm_graphkit.datasets.globals import *
from gridfm_graphkit.models.utils import ComputeBranchFlow, ComputeNodeInjection, ComputeNodeResiduals
from torch_scatter import scatter_add
import matplotlib.pyplot as plt
import seaborn as sns

class HeteroFeatureReconstructionTask(L.LightningModule):
    """
    PyTorch Lightning task for node feature reconstruction on power grid graphs.

    This task wraps a GridFM model inside a LightningModule and defines the full
    training, validation, testing, and prediction logic. It is designed to
    reconstruct masked node features from graph-structured input data, using
    datasets and normalizers provided by `gridfm-graphkit`.

    Args:
        args (NestedNamespace): Experiment configuration. Expected fields include `training.batch_size`, `optimizer.*`, etc.
        data_normalizers (list): One normalizer per dataset to (de)normalize features.

    Attributes:
        model (torch.nn.Module): model loaded via `load_model`.
        loss_fn (callable): Loss function resolved from configuration.
        batch_size (int): Training batch size. From ``args.training.batch_size``
        data_normalizers (list): Dataset-wise feature normalizers.

    Methods:
        forward(x, pe, edge_index, edge_attr, batch, mask=None):
            Forward pass with optional feature masking.
        training_step(batch):
            One training step: computes loss, logs metrics, returns loss.
        validation_step(batch, batch_idx):
            One validation step: computes losses and logs metrics.
        test_step(batch, batch_idx, dataloader_idx=0):
            Evaluate on test data, compute per-node-type MSEs, and log per-dataset metrics.
        predict_step(batch, batch_idx, dataloader_idx=0):
            Run inference and return denormalized outputs + node masks.
        configure_optimizers():
            Setup Adam optimizer and ReduceLROnPlateau scheduler.
        on_fit_start():
            Save normalization statistics at the beginning of training.
        on_test_end():
            Collect test metrics across datasets and export summary CSV reports.

    Notes:
        - Node types are distinguished using the global constants (`PQ`, `PV`, `REF`).
        - The datamodule must provide `batch.mask` for masking node features.
        - Test metrics include per-node-type RMSE for [Pd, Qd, Pg, Qg, Vm, Va].
        - Reports are saved under `<mlflow_artifacts>/test/<dataset>.csv`.

    Example:
        ```python
        model = FeatureReconstructionTask(args, data_normalizers)
        output = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
        ```
    """

    def __init__(self, args, data_normalizers):
        super().__init__()
        self.model = load_model(args=args)
        self.args = args
        self.loss_fn = get_loss_function(args)
        self.batch_size = int(args.training.batch_size)
        self.data_normalizers = data_normalizers
        self.test_outputs = []
        self.save_hyperparameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        x_dict['bus'][mask_dict['bus']] = self.args.data.mask_value
        x_dict['gen'][mask_dict['gen']] = self.args.data.mask_value
        edge_attr_dict[("bus", "connects", "bus")][mask_dict["branch"]] = self.args.data.mask_value
        return self.model(x_dict, edge_index_dict, edge_attr_dict, mask_dict)

    @rank_zero_only
    def on_fit_start(self):
        # Determine save path
        if isinstance(self.logger, MLFlowLogger):
            log_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
                "stats",
            )
        else:
            log_dir = os.path.join(self.logger.save_dir, "stats")

        os.makedirs(log_dir, exist_ok=True)
        log_stats_path = os.path.join(log_dir, "normalization_stats.txt")

        # Collect normalization stats
        with open(log_stats_path, "w") as log_file:
            for i, normalizer in enumerate(self.data_normalizers):
                log_file.write(
                    f"Data Normalizer {self.args.data.networks[i]} stats:\n{normalizer.get_stats()}\n\n",
                )

    def shared_step(self, batch):
        output = self.forward(
            x_dict=batch.x_dict,
            edge_index_dict=batch.edge_index_dict,
            edge_attr_dict=batch.edge_attr_dict,
            mask_dict=batch.mask_dict,
        )

        loss_dict = self.loss_fn(
            output,
            batch.y_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            batch.mask_dict,
            model=self.model,
        )
        return output, loss_dict

    def training_step(self, batch):
        _, loss_dict = self.shared_step(batch)
        current_lr = self.optimizer.param_groups[0]["lr"]
        metrics = {}
        metrics["Training Loss"] = loss_dict["loss"].detach()
        metrics["Learning Rate"] = current_lr
        for metric, value in metrics.items():
            self.log(
                metric,
                value,
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                on_step=False,
            )

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        _, loss_dict = self.shared_step(batch)
        loss_dict["loss"] = loss_dict["loss"].detach()
        for metric, value in loss_dict.items():
            metric_name = f"Validation {metric}"
            self.log(
                metric_name,
                value,
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_epoch=True,
                logger=True,
                on_step=False,
            )

        return loss_dict["loss"]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output, loss_dict = self.shared_step(batch)
        dataset_name = self.args.data.networks[dataloader_idx]

        self.data_normalizers[dataloader_idx].inverse_transform(batch)
        self.data_normalizers[dataloader_idx].inverse_output(output)

        branch_flow_layer = ComputeBranchFlow()
        node_injection_layer = ComputeNodeInjection()
        node_residuals_layer = ComputeNodeResiduals()

        num_bus = batch.x_dict["bus"].size(0)
        bus_edge_index = batch.edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = batch.edge_attr_dict[("bus", "connects", "bus")]
        _ , gen_to_bus_index = batch.edge_index_dict[("gen", "connected_to", "bus")]

        if self.args.data.mask_type == "opf_hetero":
            mse_PG = F.mse_loss(
                output["gen"],
                batch.y_dict["gen"],
                reduction="none",
            ).mean(dim=0)
            c0 = batch.x_dict["gen"][:, C0_H]
            c1 = batch.x_dict["gen"][:, C1_H]
            c2 = batch.x_dict["gen"][:, C2_H]
            target_pg = batch.y_dict["gen"].squeeze()
            pred_pg = output["gen"].squeeze() 
            gen_cost_gt = c0 + c1 * target_pg + c2 * target_pg**2
            gen_cost_pred = c0 + c1 * pred_pg + c2 * pred_pg**2


            gen_batch = batch.batch_dict["gen"]   # shape: [N_gen_total]

            cost_gt = scatter_add(gen_cost_gt, gen_batch, dim=0)
            cost_pred = scatter_add(gen_cost_pred, gen_batch, dim=0)

            optimality_gap = torch.mean(torch.abs((cost_pred - cost_gt) / cost_gt * 100))
        
        agg_gen_on_bus = scatter_add(batch.y_dict["gen"], gen_to_bus_index, dim=0, dim_size=num_bus)
        #output_agg = torch.cat([batch.y_dict["bus"], agg_gen_on_bus], dim=1)
        target = torch.stack([batch.y_dict['bus'][:, VM_H], batch.y_dict['bus'][:, VA_H], agg_gen_on_bus.squeeze(), batch.y_dict['bus'][:, QG_H]], dim=1)
        
        # UN-COMMENT THIS TO CHECK PBE ON GROUND TRUTH
        #output["bus"] = target 

        Pft, Qft = branch_flow_layer(output["bus"], bus_edge_index, bus_edge_attr)
        # Compute branch termal limits violations
        Sft = torch.sqrt(Pft**2 + Qft**2)  # apparent power flow per branch
        branch_thermal_limits = bus_edge_attr[:, RATE_A]
        branch_thermal_excess = F.relu(Sft - branch_thermal_limits)

        num_edges = bus_edge_index.size(1)
        half_edges = num_edges // 2
        forward_excess = branch_thermal_excess[:half_edges]
        reverse_excess = branch_thermal_excess[half_edges:]


        mean_thermal_violation_forward = torch.mean(forward_excess)
        mean_thermal_violation_reverse = torch.mean(reverse_excess)

        # Compute branch angle difference violation
        angle_min = bus_edge_attr[:, ANG_MIN]
        angle_max = bus_edge_attr[:, ANG_MAX]

        bus_angles = output["bus"][:, VA_OUT]  # in degrees
        from_bus = bus_edge_index[0]
        to_bus = bus_edge_index[1]
        angle_diff = torch.abs(bus_angles[from_bus] - bus_angles[to_bus])
 
        angle_excess_low = F.relu(angle_min - angle_diff)   # violation if too small
        angle_excess_high = F.relu(angle_diff - angle_max)  # violation if too large
        branch_angle_violation_mean = torch.mean(angle_excess_low + angle_excess_high) * 180.0 / torch.pi


        P_in, Q_in = node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
        residual_P, residual_Q = node_residuals_layer(P_in, Q_in, output["bus"], batch.x_dict["bus"])

        final_residual_real_bus = torch.mean(torch.abs(residual_P))
        final_residual_imag_bus = torch.mean(torch.abs(residual_Q))

        loss_dict["Active Power Loss"] = final_residual_real_bus.detach()
        loss_dict["Reactive Power Loss"] = final_residual_imag_bus.detach()

        mask_PQ = batch.mask_dict["PQ"]  # PQ buses
        mask_PV = batch.mask_dict["PV"]  # PV buses
        mask_REF = batch.mask_dict["REF"] # Reference buses

        self.test_outputs.append({
            "dataset": dataset_name,
            "pred": output["bus"].detach().cpu(),
            "target": target.detach().cpu(),
            "mask_PQ": mask_PQ.cpu(),
            "mask_PV": mask_PV.cpu(),
            "mask_REF": mask_REF.cpu(),
        })

        mse_PQ = F.mse_loss(
            output["bus"][mask_PQ],
            target[mask_PQ],
            reduction="none",
        )
        mse_PV = F.mse_loss(
            output["bus"][mask_PV],
            target[mask_PV],
            reduction="none",
        )
        mse_REF = F.mse_loss(
            output["bus"][mask_REF],
            target[mask_REF],
            reduction="none",
        )

        mse_PQ = mse_PQ.mean(dim=0)
        mse_PV = mse_PV.mean(dim=0)
        mse_REF = mse_REF.mean(dim=0)

        if self.args.data.mask_type == "opf_hetero":
            loss_dict["Opt gap"] = optimality_gap
            loss_dict["MSE PG"] = mse_PG[PG_H]

        loss_dict["Branch termal violation from"] = mean_thermal_violation_forward
        loss_dict["Branch termal violation to"] = mean_thermal_violation_reverse
        loss_dict["Branch voltage angle difference violations"] = branch_angle_violation_mean

        loss_dict["MSE PQ nodes - PG"] = mse_PQ[PG_OUT]
        loss_dict["MSE PV nodes - PG"] = mse_PV[PG_OUT]
        loss_dict["MSE REF nodes - PG"] = mse_REF[PG_OUT]

        loss_dict["MSE PQ nodes - QG"] = mse_PQ[QG_OUT]
        loss_dict["MSE PV nodes - QG"] = mse_PV[QG_OUT]
        loss_dict["MSE REF nodes - QG"] = mse_REF[QG_OUT]

        loss_dict["MSE PQ nodes - VM"] = mse_PQ[VM_OUT]
        loss_dict["MSE PV nodes - VM"] = mse_PV[VM_OUT]
        loss_dict["MSE REF nodes - VM"] = mse_REF[VM_OUT]

        loss_dict["MSE PQ nodes - VA"] = mse_PQ[VA_OUT]
        loss_dict["MSE PV nodes - VA"] = mse_PV[VA_OUT]
        loss_dict["MSE REF nodes - VA"] = mse_REF[VA_OUT]

        loss_dict["Test loss"] = loss_dict.pop("loss").detach()
        for metric, value in loss_dict.items():
            metric_name = f"{dataset_name}/{metric}"
            self.log(
                metric_name,
                value,
                batch_size=batch.num_graphs,
                add_dataloader_idx=False,
                sync_dist=True,
                logger=False,
            )
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError
        output, _ = self.shared_step(batch)
        self.data_normalizers[dataloader_idx].inverse_output(output)

        mask_PQ = batch.x_dict["bus"][:, PQ_H] == 1  # PQ buses
        mask_PV = batch.x_dict["bus"][:, PV_H] == 1  # PV buses
        mask_REF = batch.x_dict["bus"][:, REF_H] == 1  # Reference buses

        # Count buses and generate per-node scenario_id
        bus_counts = batch.batch.unique(return_counts=True)[1]
        scenario_ids = batch.scenario_id  # shape: [num_graphs]
        scenario_per_node = torch.cat(
            [
                torch.full((count,), sid, dtype=torch.int32)
                for count, sid in zip(bus_counts, scenario_ids)
            ],
        )

        bus_numbers = np.concatenate([np.arange(count.item()) for count in bus_counts])

        return {
            "output": output_denorm.cpu().numpy(),
            "mask_PQ": mask_PQ,
            "mask_PV": mask_PV,
            "mask_REF": mask_REF,
            "scenario_id": scenario_per_node,
            "bus_number": bus_numbers,
        }

    @rank_zero_only
    def on_test_end(self):
        if isinstance(self.logger, MLFlowLogger):
            artifact_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
            )
        else:
            artifact_dir = self.logger.save_dir

        final_metrics = self.trainer.callback_metrics
        grouped_metrics = {}

        for full_key, value in final_metrics.items():
            try:
                value = value.item()
            except AttributeError:
                pass

            if "/" in full_key:
                dataset_name, metric = full_key.split("/", 1)
                if dataset_name not in grouped_metrics:
                    grouped_metrics[dataset_name] = {}
                grouped_metrics[dataset_name][metric] = value

        for dataset, metrics in grouped_metrics.items():
            # RMSE metrics
            rmse_PQ = [
                metrics.get(f"MSE PQ nodes - {label}", float("nan")) ** 0.5
                for label in ["PG", "QG", "VM", "VA"]
            ]
            rmse_PV = [
                metrics.get(f"MSE PV nodes - {label}", float("nan")) ** 0.5
                for label in ["PG", "QG", "VM", "VA"]
            ]
            rmse_REF = [
                metrics.get(f"MSE REF nodes - {label}", float("nan")) ** 0.5
                for label in ["PG", "QG", "VM", "VA"]
            ]

            # Residuals and generator metrics
            avg_active_res = metrics.get("Active Power Loss", " ")
            avg_reactive_res = metrics.get("Reactive Power Loss", " ")
            rmse_gen = metrics.get("MSE PG", 0) ** 0.5
            optimality_gap = metrics.get("Opt gap", " ")
            branch_thermal_violation_from = metrics.get("Branch termal violation from", " ")
            branch_thermal_violation_to = metrics.get("Branch termal violation to", " ")
            branch_angle_violation = metrics.get("Branch voltage angle difference violations", " ")

            # --- Main RMSE metrics file ---
            data_main = {
                "Metric": ["RMSE-PQ", "RMSE-PV", "RMSE-REF"],
                "Pg (MW)": [rmse_PQ[0], rmse_PV[0], rmse_REF[0]],
                "Qg (MVar)": [rmse_PQ[1], rmse_PV[1], rmse_REF[1]],
                "Vm (p.u.)": [rmse_PQ[2], rmse_PV[2], rmse_REF[2]],
                "Va (radians)": [rmse_PQ[3], rmse_PV[3], rmse_REF[3]],
            }
            df_main = pd.DataFrame(data_main)

            # --- Residuals / generator metrics file ---
            data_residuals = {
                "Metric": [
                    "Avg. active res. (MW)",
                    "Avg. reactive res. (MVar)",
                    "RMSE PG generators (MW)",
                    "Mean optimality gap (%)", 
                    "Mean branch termal violation from (MVA)", 
                    "Mean branch termal violation to (MVA)", 
                    "Mean branch angle difference violation (radians)",
                ],
                "Value": [avg_active_res, avg_reactive_res, rmse_gen, optimality_gap, branch_thermal_violation_from, branch_thermal_violation_to, branch_angle_violation]
            }
            df_residuals = pd.DataFrame(data_residuals)

            # --- Save CSVs ---
            test_dir = os.path.join(artifact_dir, "test")
            os.makedirs(test_dir, exist_ok=True)

            main_csv_path = os.path.join(test_dir, f"{dataset}_RMSE.csv")
            residuals_csv_path = os.path.join(test_dir, f"{dataset}_metrics.csv")

            df_main.to_csv(main_csv_path, index=False)
            df_residuals.to_csv(residuals_csv_path, index=False)
        plot_dir = os.path.join(artifact_dir, "test_plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Concatenate predictions and targets across all batches
        all_preds = torch.cat([d["pred"] for d in self.test_outputs])
        all_targets = torch.cat([d["target"] for d in self.test_outputs])
        all_masks = {
            "PQ": torch.cat([d["mask_PQ"] for d in self.test_outputs]),
            "PV": torch.cat([d["mask_PV"] for d in self.test_outputs]),
            "REF": torch.cat([d["mask_REF"] for d in self.test_outputs]),
        }

        feature_labels = ["Vm", "Va", "Pg", "Qg"]

        # Create correlation plots per node type
        for node_type, mask in all_masks.items():
            preds = all_preds[mask]
            targets = all_targets[mask]
            if preds.numel() == 0:
                continue

            num_features = preds.shape[1]
            fig, axes = plt.subplots(2, 2, figsize=(15, 8))
            axes = axes.flatten()

            for i, (ax, label) in enumerate(zip(axes, feature_labels)):
                x = targets[:, i].numpy().flatten()
                y = preds[:, i].numpy().flatten()
                sns.scatterplot(
                    x=x,
                    y=y,
                    s=6,          # smaller dots
                    alpha=0.4,    # transparency
                    ax=ax,
                    edgecolor=None
                )
                # Add y=x reference line
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.0, alpha=0.7)

                corr = np.corrcoef(x, y)[0, 1]
                ax.set_title(f"{node_type} – {label}\nR² = {corr**2:.3f}")
                ax.set_xlabel("Target")
                ax.set_ylabel("Prediction")

            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"correlation_{node_type}.png"), dpi=300)
            plt.close(fig)

        self.test_outputs.clear()

    def configure_optimizers(self):
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=self.args.optimizer.learning_rate,
        #     betas=(self.args.optimizer.beta1, self.args.optimizer.beta2),
        # )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.optimizer.learning_rate,
            betas=(self.args.optimizer.beta1, self.args.optimizer.beta2),
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.args.optimizer.lr_decay,
            patience=self.args.optimizer.lr_patience,
        )
        config_optim = {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "Validation loss",
                "reduce_on_plateau": True,
            },
        }
        return config_optim
