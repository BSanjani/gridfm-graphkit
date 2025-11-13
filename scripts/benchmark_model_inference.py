import torch
from torch_geometric.loader import DataLoader
import csv
from torch_geometric.data import Data
from gridfm_graphkit.io.param_handler import NestedNamespace, load_model
import argparse
import yaml

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="GridFM Benchmarking Script")
parser.add_argument('--config', type=str, required=True, help='Path to config file')
parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes in synthetic graph')
parser.add_argument('--num_edges', type=int, required=True, help='Number of edges in synthetic graph')
parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file')
args = parser.parse_args()

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(args.config, "r") as f:
    base_config = yaml.safe_load(f)

config_args = NestedNamespace(**base_config)

model = load_model(config_args)

model = model.to(device).eval()

NUM_NODES = args.num_nodes
NUM_EDGES = args.num_edges
NODE_FEATURES = 9
EDGE_FEATURES = 2

# --- Generate Random Graph ---
# Random undirected edges (avoid self-loops)
row = torch.randint(0, NUM_NODES, (NUM_EDGES,))
col = torch.randint(0, NUM_NODES, (NUM_EDGES,))
edge_index = torch.stack([row, col], dim=0)

# Random features
x = torch.randn(NUM_NODES, NODE_FEATURES)
edge_attr = torch.randn(NUM_EDGES, EDGE_FEATURES)

# Fake positional encodings and batch vector
pe = torch.randn(NUM_NODES, 20)
batch_vec = torch.zeros(NUM_NODES, dtype=torch.long)  # single graph per batch

# Build PyG Data object
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pe=pe, batch=batch_vec)

# --- Batch Sizes ---
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512]
samples = 1024

# --- CSV Output File Path ---
csv_file_path = args.output_csv

# --- Benchmark Function ---
def benchmark():
    with open(csv_file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["batch_size", "time(ms)"])
        model.eval()

        with torch.no_grad():
            for batch_size in batch_sizes:
                # Create synthetic batch (duplicated graph)
                data_list = [graph.clone() for _ in range(batch_size)]
                loader = DataLoader(data_list, batch_size=batch_size)

                batch = next(iter(loader)).to(device)

                num_iters = samples // batch_size
                print(f"Running with batch size {batch_size} for {num_iters} iterations...")

                # Warm-up runs to stabilize GPU performance
                for _ in range(10):
                    _ = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)

                # Synchronize before timing
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                for _ in range(num_iters):
                    _ = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                end.record()

                torch.cuda.synchronize()
                total_time = start.elapsed_time(end) / samples
                print(f"Time taken with batch size {batch_size}: {total_time:.4f} milliseconds")

                writer.writerow([batch_size, total_time])
                csv_file.flush()

if __name__ == "__main__":
    print("Benchmark starting...")
    benchmark()
    print("Benchmark completed.")