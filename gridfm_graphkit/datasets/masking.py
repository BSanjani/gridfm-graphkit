from gridfm_graphkit.datasets.globals import *
from gridfm_graphkit.io.registries import MASKING_REGISTRY

import torch
from torch_geometric.transforms import BaseTransform

@MASKING_REGISTRY.register("none")
class AddIdentityMask(BaseTransform):
    """Creates an identity mask, and adds it as a `mask` attribute.

    The mask is generated such that every entry is False, so no masking is actually applied
    """

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        # Generate an identity mask
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        # Add the mask to the data object
        data.mask = mask

        return data


@MASKING_REGISTRY.register("rnd")
class AddRandomMask(BaseTransform):
    """Creates a random mask, and adds it as a `mask` attribute.

    The mask is generated such that each entry is `True` with probability
    `mask_ratio` and `False` otherwise.
    """

    def __init__(self, args):
        super().__init__()
        self.mask_dim = args.data.mask_dim
        self.mask_ratio = args.data.mask_ratio

    def forward(self, data):
        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate a random mask
        mask = torch.rand(data.x.size(0), self.mask_dim) < self.mask_ratio

        # Add the mask to the data object
        data.mask = mask

        return data

class AddPFHeteroMask(BaseTransform):
    """Creates masks for a heterogeneous power flow graph."""

    def __init__(self):
        super().__init__()

    def forward(self, data):
        bus_x = data.x_dict["bus"]
        gen_x = data.x_dict["gen"]

        # Identify bus types
        mask_PQ = bus_x[:, PQ_H] == 1
        mask_PV = bus_x[:, PV_H] == 1
        mask_REF = bus_x[:, REF_H] == 1

        # Initialize mask tensors
        mask_bus = torch.zeros_like(bus_x, dtype=torch.bool)
        mask_gen = torch.zeros_like(gen_x, dtype=torch.bool)

        mask_bus[:, MIN_VM_H] = True
        mask_bus[:, MAX_VM_H] = True
        mask_bus[:, MIN_QG_H] = True
        mask_bus[:, MAX_QG_H] = True
        mask_bus[:, VN_KV] = True

        mask_gen[:, MIN_PG] = True
        mask_gen[:, MAX_PG] = True
        mask_gen[:, C0_H] = True
        mask_gen[:, C1_H] = True
        mask_gen[:, C2_H] = True


        # --- PQ buses ---
        mask_bus[mask_PQ, VM_H] = True
        mask_bus[mask_PQ, VA_H] = True

        # --- PV buses ---
        mask_bus[mask_PV, VA_H] = True
        mask_bus[mask_PV, QG_H] = True

        # --- REF buses ---
        mask_bus[mask_REF, VM_H] = True
        mask_bus[mask_REF, QG_H] = True
        # --- Generators connected to REF buses ---
        gen_bus_edges = data.edge_index_dict[("gen", "connected_to", "bus")]
        gen_indices, bus_indices = gen_bus_edges
        ref_gens = gen_indices[mask_REF[bus_indices]]
        mask_gen[ref_gens, PG_H] = True

        mask_branch = torch.zeros_like(data.edge_attr_dict[("bus", "connects", "bus")], dtype=torch.bool)
        mask_branch[:, P_E] = True
        mask_branch[:, Q_E] = True 

        data.mask_dict = {
            "bus": mask_bus,
            "gen": mask_gen,
            "branch": mask_branch,
            "PQ": mask_PQ,
            "PV": mask_PV,
            "REF": mask_REF,
        }

        return data

class AddOPFHeteroMask(BaseTransform):
    """Creates masks for a heterogeneous power flow graph."""

    def __init__(self):
        super().__init__()

    def forward(self, data):
        bus_x = data.x_dict["bus"]
        gen_x = data.x_dict["gen"]

        # Identify bus types
        mask_PQ = bus_x[:, PQ_H] == 1
        mask_PV = bus_x[:, PV_H] == 1
        mask_REF = bus_x[:, REF_H] == 1

        # Initialize mask tensors
        mask_bus = torch.zeros_like(bus_x, dtype=torch.bool)
        mask_gen = torch.zeros_like(gen_x, dtype=torch.bool)


        # --- PQ buses ---
        mask_bus[mask_PQ, VM_H] = True
        mask_bus[mask_PQ, VA_H] = True

        # --- PV buses ---
        mask_bus[mask_PV, VA_H] = True
        mask_bus[mask_PV, VM_H] = True
        mask_bus[mask_PV, QG_H] = True

        # --- REF buses ---
        mask_bus[mask_REF, QG_H] = True
        mask_bus[mask_REF, VM_H] = True

        mask_gen[:, PG_H] = True

        mask_branch = torch.zeros_like(data.edge_attr_dict[("bus", "connects", "bus")], dtype=torch.bool)
        mask_branch[:, P_E] = True  
        mask_branch[:, Q_E] = True 

        data.mask_dict = {
            "bus": mask_bus,
            "gen": mask_gen,
            "branch": mask_branch,
            "PQ": mask_PQ,
            "PV": mask_PV,
            "REF": mask_REF,
        }

        return data

@MASKING_REGISTRY.register("pf")
class AddPFMask(BaseTransform):
    """Creates a mask according to the power flow problem and assigns it as a `mask` attribute."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate masks for each type of node
        mask_PQ = data.x[:, PQ] == 1  # PQ buses
        mask_PV = data.x[:, PV] == 1  # PV buses
        mask_REF = data.x[:, REF] == 1  # Reference buses

        # Initialize the mask tensor with False values
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        mask[mask_PQ, VM] = True  # Mask Vm for PQ buses
        mask[mask_PQ, VA] = True  # Mask Va for PQ buses

        mask[mask_PV, QG] = True  # Mask Qg for PV buses
        mask[mask_PV, VA] = True  # Mask Va for PV buses

        mask[mask_REF, PG] = True  # Mask Pg for REF buses
        mask[mask_REF, QG] = True  # Mask Qg for REF buses

        # Attach the mask to the data object
        data.mask = mask

        return data


@MASKING_REGISTRY.register("opf")
class AddOPFMask(BaseTransform):
    """Creates a mask according to the optimal power flow problem and assigns it as a `mask` attribute."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate masks for each type of node
        mask_PQ = data.x[:, PQ] == 1  # PQ buses
        mask_PV = data.x[:, PV] == 1  # PV buses
        mask_REF = data.x[:, REF] == 1  # Reference buses

        # Initialize the mask tensor with False values
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        mask[mask_PQ, VM] = True  # Mask Vm for PQ
        mask[mask_PQ, VA] = True  # Mask Va for PQ

        mask[mask_PV, PG] = True  # Mask Pg for PV
        mask[mask_PV, QG] = True  # Mask Qg for PV
        mask[mask_PV, VM] = True  # Mask Vm for PV
        mask[mask_PV, VA] = True  # Mask Va for PV

        mask[mask_REF, PG] = True  # Mask Pg for REF
        mask[mask_REF, QG] = True  # Mask Qg for REF
        mask[mask_REF, VM] = True  # Mask Vm for REF
        mask[mask_REF, VA] = True  # Mask Va for REF

        # Attach the mask to the data object
        data.mask = mask

        return data
