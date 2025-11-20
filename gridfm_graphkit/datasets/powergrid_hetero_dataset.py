from gridfm_graphkit.datasets.normalizers import Normalizer, BaseMVANormalizer

import os.path as osp
import os
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.globals import *



class HeteroGridDatasetDisk(Dataset):
    """
    A PyTorch Geometric `Dataset` for power grid data stored on disk.
    This dataset reads node and edge CSV files, applies normalization,
    and saves each graph separately on disk as a processed file.
    Data is loaded from disk lazily on demand.

    Args:
        root (str): Root directory where the dataset is stored.
        norm_method (str): Identifier for normalization method (e.g., "minmax", "standard").
        node_normalizer (Normalizer): Normalizer used for node features.
        edge_normalizer (Normalizer): Normalizer used for edge features.
        pe_dim (int): Length of the random walk used for positional encoding.
        mask_dim (int, optional): Number of features per-node that could be masked.
        transform (callable, optional): Transformation applied at runtime.
        pre_transform (callable, optional): Transformation applied before saving to disk.
        pre_filter (callable, optional): Filter to determine which graphs to keep.
    """

    def __init__(
        self,
        root: str,
        norm_method: str,
        node_normalizer: Normalizer,
        edge_normalizer: Normalizer,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.norm_method = norm_method
        self.node_normalizer = node_normalizer
        self.edge_normalizer = edge_normalizer
        self.length = None

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load normalization stats if available
        node_stats_path = osp.join(
            self.processed_dir,
            f"node_stats_{self.norm_method}.pt",
        )
        edge_stats_path = osp.join(
            self.processed_dir,
            f"edge_stats_{self.norm_method}.pt",
        )
        if osp.exists(node_stats_path) and osp.exists(edge_stats_path):
            self.node_stats = torch.load(node_stats_path, weights_only=False)
            self.edge_stats = torch.load(edge_stats_path, weights_only=False)
            self.node_normalizer.fit_from_dict(self.node_stats)
            self.edge_normalizer.fit_from_dict(self.edge_stats)

    # @property
    # def raw_file_names(self):
    #     return ["bus_data.csv", "gen_data.csv", "y_bus_data.csv"]
    @property
    def raw_file_names(self):
        return ["bus_data.parquet", "gen_data.parquet", "branch_data.parquet"]

    @property
    def processed_done_file(self):
        return f"processed_no_y_bus_{self.norm_method}.done"

    @property
    def processed_file_names(self):
        return [self.processed_done_file]

    def download(self):
        pass

    
    def process(self):
        print("Loading data...")
        bus_data = pd.read_parquet(osp.join(self.raw_dir, "bus_data.parquet"))
        gen_data = pd.read_parquet(osp.join(self.raw_dir, "gen_data.parquet"))
        branch_data = pd.read_parquet(osp.join(self.raw_dir, "branch_data.parquet"))

        agg_gen = gen_data.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
        bus_data = bus_data.merge(agg_gen, on=["scenario", "bus"], how="left").fillna(0)

        print("Fit normalizers...")
        self.node_stats = self.node_normalizer.fit(bus_data=bus_data, gen_data=gen_data)
        self.edge_stats = self.edge_normalizer.fit(baseMVA=self.node_normalizer.baseMVA)
        node_stats_path = osp.join(
            self.processed_dir,
            f"node_stats_{self.norm_method}.pt",
        )
        edge_stats_path = osp.join(
            self.processed_dir,
            f"edge_stats_{self.norm_method}.pt",
        )
        torch.save(self.node_stats, node_stats_path)
        torch.save(self.edge_stats, edge_stats_path)

        bus_features = ["Pd", "Qd", "Qg", "Vm", "Va", "PQ", "PV", "REF", "min_vm_pu", "max_vm_pu", "min_q_mvar", "max_q_mvar", "GS", "BS"]
        gen_features = ["p_mw", "min_p_mw", "max_p_mw", "cp0_eur" , "cp1_eur_per_mw" , "cp2_eur_per_mw2"]
        common_branch_features = ["tap", "ang_min", "ang_max", "rate_a"]
        forward_branch_features = ["pf", "qf", "Yff_r", "Yff_i", "Yft_r", "Yft_i"] + common_branch_features
        reverse_branch_features = ["pt", "qt", "Ytt_r", "Ytt_i", "Ytf_r", "Ytf_i"] + common_branch_features
        
        print("Normalize data...")
        bus_tensor, gen_tensor = self.node_normalizer.transform(
            bus_data=torch.tensor(bus_data[bus_features].values, dtype=torch.float),
            gen_data=torch.tensor(gen_data[gen_features].values, dtype=torch.float)
        )
        bus_data[bus_features] = bus_tensor.numpy()
        gen_data[gen_features] = gen_tensor.numpy()

        # Group by scenario
        bus_groups = bus_data.groupby("scenario")
        gen_groups = gen_data.groupby("scenario")
        branch_groups = branch_data.groupby("scenario")

        # Process each scenario
        print("Save data...")
        for scenario in tqdm(bus_data["scenario"].unique(), desc="Processing scenarios"):
            if scenario not in gen_groups.groups or scenario not in branch_groups.groups:
                raise ValueError

            data = HeteroData()

            # Bus nodes
            bus_df = bus_groups.get_group(scenario)
            data["bus"].x = torch.tensor(bus_df[bus_features].values, dtype=torch.float)

            # Generator nodes
            gen_df = gen_groups.get_group(scenario).reset_index()
            data["gen"].x = torch.tensor(gen_df[gen_features].values, dtype=torch.float)
            gen_df["gen_index"] = gen_df.index  # Use actual index as generator ID

            #data["bus"].x , data["gen"].x = self.node_normalizer.transform(bus_data=data["bus"].x , gen_data=data["gen"].x)
            
            data["bus"].y = data["bus"].x[: , :(VA_H+1)].clone()
            data["gen"].y = data["gen"].x[: , :(PG_H+1)].clone()

            # Bus-Bus edges
            branch_df = branch_groups.get_group(scenario)
            branch_df = branch_df[branch_df["br_status"] != 0].reset_index(drop=True)

            forward_edges = torch.tensor(branch_df[["from_bus","to_bus"]].values.T, dtype=torch.long)
            forward_edge_attr = self.edge_normalizer.transform(edge_data=torch.tensor(branch_df[forward_branch_features].values, dtype=torch.float))
            
            reverse_edges = torch.tensor(branch_df[["to_bus","from_bus"]].values.T, dtype=torch.long)
            reverse_edge_attr = self.edge_normalizer.transform(edge_data=torch.tensor(branch_df[reverse_branch_features].values, dtype=torch.float))

            edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
            edge_attr = torch.cat([forward_edge_attr, reverse_edge_attr], dim=0)

            forward_targets = torch.tensor(branch_df[["pf", "qf"]].values, dtype=torch.float)
            reverse_targets = torch.tensor(branch_df[["pt", "qt"]].values, dtype=torch.float)
            edge_y = torch.cat([forward_targets, reverse_targets], dim=0)

            data["bus", "connects", "bus"].edge_index = edge_index
            data["bus", "connects", "bus"].edge_attr = edge_attr
            data["bus", "connects", "bus"].y = edge_y

            #data["bus", "connects", "bus"].edge_attr = self.edge_normalizer.transform(edge_data=data["bus", "connects", "bus"].edge_attr)

            # Gen-Bus and Bus-Gen edges
            data["gen", "connected_to", "bus"].edge_index = torch.tensor(
                gen_df[["gen_index", "bus"]].values.T, dtype=torch.long
            )
            data["bus", "connected_to", "gen"].edge_index = torch.tensor(
                gen_df[["bus", "gen_index"]].values.T, dtype=torch.long
            )
                

            # Save graph
            torch.save(data, osp.join(self.processed_dir, f"data_no_y_bus_{self.norm_method}_index_{scenario}.pt"))

        with open(osp.join(self.processed_dir, self.processed_done_file), "w") as f:
            f.write("done")
            
    def len(self):
        if self.length is None:
            files = [
                f
                for f in os.listdir(self.processed_dir)
                if f.startswith(
                    f"data_no_y_bus_{self.norm_method}_index_",
                )
                and f.endswith(".pt")
            ]
            self.length = len(files)
        return self.length

    def get(self, idx):
        file_name = osp.join(
            self.processed_dir,
            f"data_no_y_bus_{self.norm_method}_index_{idx}.pt",
        )
        if not osp.exists(file_name):
            raise IndexError(f"Data file {file_name} does not exist.")
        data = torch.load(file_name, weights_only=False)
        return data

    def change_transform(self, new_transform):
        """
        Temporarily switch to a new transform function, used when evaluating different tasks.

        Args:
            new_transform (Callable): The new transform to use.
        """
        self.original_transform = self.transform
        self.transform = new_transform

    def reset_transform(self):
        """
        Reverts the transform to the original one set during initialization, usually called after the evaluation step.
        """
        if self.original_transform is None:
            raise ValueError(
                "The original transform is None or the function change_transform needs to be called before",
            )
        self.transform = self.original_transform
