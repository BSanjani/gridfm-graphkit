from torch_geometric.transforms import Compose
from gridfm_graphkit.datasets.transforms import RemoveInactiveBranches, RemoveInactiveGenerators, ApplyMasking
from gridfm_graphkit.datasets.masking import AddOPFHeteroMask, AddPFHeteroMask
from gridfm_graphkit.io.registries import TRANSFORM_REGISTRY

@TRANSFORM_REGISTRY.register("PowerFlow")
class PowerFlowTransforms(Compose):
    def __init__(
        self, args
    ):
        transforms = []

        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())
        transforms.append(AddPFHeteroMask())
        transforms.append(ApplyMasking(args=args))

        # Pass the list of transforms to Compose
        super().__init__(transforms)

@TRANSFORM_REGISTRY.register("OptimalPowerFlow")
class OptimalPowerFlowTransforms(Compose):
    def __init__(
        self, args
    ):
        transforms = []

        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())
        transforms.append(AddOPFHeteroMask())
        transforms.append(ApplyMasking(args=args))

        # Pass the list of transforms to Compose
        super().__init__(transforms)