import torch
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.io.param_handler import NestedNamespace
import yaml
from gridfm_graphkit.models.utils import ComputeBranchFlow
from gridfm_graphkit.datasets.globals import VM_H, VA_H, P_E, Q_E


def test_edge_flows():
    data = torch.load(
        "/dccstor/gridfm/opf_data/case14_ieee/processed/data_index_0.pt",
        weights_only=False,
    )

    node_stats = torch.load(
        "/dccstor/gridfm/opf_data/case14_ieee/processed/data_stats_HeteroDataMVANormalizer.pt",
        weights_only=False,
    )
    with open("examples/config/HGNS_case14_SE.yaml", "r") as f:
        args = yaml.safe_load(f)
    args = NestedNamespace(**args)
    normalizer = HeteroDataMVANormalizer(args)
    normalizer.fit_from_dict(node_stats)
    normalizer.transform(data)

    bus_edge_index = data[("bus", "connects", "bus")].edge_index
    bus_edge_attr = data[("bus", "connects", "bus")].edge_attr
    branch_flow_layer = ComputeBranchFlow()

    Pft, Qft = branch_flow_layer(
        data["bus"].x[:, [VM_H, VA_H]],
        bus_edge_index,
        bus_edge_attr,
    )

    assert torch.isclose(Pft, bus_edge_attr[:, P_E], atol=1e-4).all()
    assert torch.isclose(Qft, bus_edge_attr[:, Q_E], atol=1e-4).all()

    print("hi")


if __name__ == "__main__":
    test_edge_flows()
