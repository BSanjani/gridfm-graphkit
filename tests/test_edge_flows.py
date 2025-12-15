import torch
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.io.param_handler import NestedNamespace
import yaml
from gridfm_graphkit.models.utils import ComputeBranchFlow, ComputeNodeInjection, PhysicsDecoderSE
from gridfm_graphkit.training.loss import LossPerDim
from gridfm_graphkit.datasets.globals import VM_H, VA_H, P_E, Q_E, VM_OUT, VA_OUT, PG_OUT, QG_OUT, QD_H, QG_H


def test_edge_flows():
    data = torch.load(
        "/dccstor/gridfm/powermodels_data/v2/opf/case118_ieee/processed/data_index_0.pt",
        weights_only=False,
    )

    node_stats = torch.load(
        "/dccstor/gridfm/powermodels_data/v2/opf/case118_ieee/processed/data_stats_HeteroDataMVANormalizer.pt",
        weights_only=False,
    )
    with open("examples/config/HGNS_OPF_datakit_case118.yaml", "r") as f:
        args = yaml.safe_load(f)
    args = NestedNamespace(**args)
    normalizer = HeteroDataMVANormalizer(args)
    normalizer.fit_from_dict(node_stats)
    normalizer.transform(data)

    bus_edge_index = data[("bus", "connects", "bus")].edge_index
    bus_edge_attr = data[("bus", "connects", "bus")].edge_attr
    branch_flow_layer = ComputeBranchFlow()
    node_injection_layer = ComputeNodeInjection()
    physics_decoder = PhysicsDecoderSE()

    Pft, Qft = branch_flow_layer(
        data["bus"].x[:, [VM_H, VA_H]],
        bus_edge_index,
        bus_edge_attr,
    )

    assert torch.isclose(Pft, bus_edge_attr[:, P_E], atol=1e-4).all()
    assert torch.isclose(Qft, bus_edge_attr[:, Q_E], atol=1e-4).all()

    P_in, Q_in = node_injection_layer(
        Pft,
        Qft,
        bus_edge_index,
        data['bus'].x.size(0),
    )

    output_temp = physics_decoder(
        P_in,
        Q_in,
        data['bus'].y[:, [VM_H, VA_H]],
        data['bus'].x,
        None,
        None,
    )

    pred_dict = {"bus": output_temp, "gen": data["gen"].y}
    target_dict = {"bus": data['bus'].y, "gen": data["gen"].y}
    edge_index = {("gen", "connected_to", "bus"): data[("gen", "connected_to", "bus")].edge_index}
    assert LossPerDim(NestedNamespace(**{"loss_str": "MAE", "dim": "VM"}), None)(pred_dict, target_dict, edge_index, None, None)['loss'] < 1e-4
    assert LossPerDim(NestedNamespace(**{"loss_str": "MAE", "dim": "VA"}), None)(pred_dict, target_dict, edge_index, None, None)['loss'] < 1e-4
    assert LossPerDim(NestedNamespace(**{"loss_str": "MAE", "dim": "P_in"}), None)(pred_dict, target_dict, edge_index, None, None)['loss'] < 1e-4
    assert LossPerDim(NestedNamespace(**{"loss_str": "MAE", "dim": "Q_in"}), None)(pred_dict, target_dict, edge_index, None, None)['loss'] < 1e-4



    print("hi")


if __name__ == "__main__":
    test_edge_flows()
