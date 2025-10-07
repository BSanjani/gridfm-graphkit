from gridfm_graphkit.training.utils import compute_node_residuals
import torch.nn.functional as F
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all custom loss functions.
    """

    @abstractmethod
    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        """
        Compute the loss.

        Parameters:
        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.
        - model: Optional model reference for accessing internal states.

        Returns:
        - A dictionary with the total loss and any additional metrics.
        """
        pass


class MaskedMSELoss(BaseLoss):
    """
    Mean Squared Error loss computed only on masked elements.
    """

    def __init__(self, reduction="mean"):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
        return {"loss": loss, "Masked MSE loss": loss.detach()}


class MSELoss(BaseLoss):
    """Standard Mean Squared Error loss."""

    def __init__(self, reduction="mean"):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return {"loss": loss, "MSE loss": loss.detach()}


class SCELoss(BaseLoss):
    """Scaled Cosine Error Loss with optional masking and normalization."""

    def __init__(self, alpha=3):
        super(SCELoss, self).__init__()
        self.alpha = alpha

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        if mask is not None:
            pred = F.normalize(pred[mask], p=2, dim=-1)
            target = F.normalize(target[mask], p=2, dim=-1)
        else:
            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

        loss = ((1 - (pred * target).sum(dim=-1)).pow(self.alpha)).mean()

        return {
            "loss": loss,
            "SCE loss": loss.detach(),
        }


class PBELoss(BaseLoss):
    def __init__(self, visualization=False):
        super().__init__()
        self.visualization = visualization

    def forward(self, pred, target, edge_index, edge_attr, mask, model=None):
        residual_complex = compute_node_residuals(
            pred,
            edge_index,
            edge_attr,
            mask,
            target,
        )

        loss = torch.mean(torch.abs(residual_complex))
        real_loss = torch.mean(torch.abs(torch.real(residual_complex)))
        imag_loss = torch.mean(torch.abs(torch.imag(residual_complex)))

        result = {
            "loss": loss,
            "Power loss": loss.detach(),
            "Active Power Loss": real_loss.detach(),
            "Reactive Power Loss": imag_loss.detach(),
        }

        if self.visualization:
            result.update(
                {
                    "Nodal Active Power Loss": torch.abs(
                        torch.real(residual_complex),
                    ),
                    "Nodal Reactive Power Loss": torch.abs(
                        torch.imag(residual_complex),
                    ),
                },
            )

        return result


class MixedLoss(BaseLoss):
    """
    Combines multiple loss functions with weighted sum.

    Args:
        loss_functions (list[nn.Module]): List of loss functions.
        weights (list[float]): Corresponding weights for each loss function.
    """

    def __init__(self, loss_functions, weights):
        super(MixedLoss, self).__init__()

        if len(loss_functions) != len(weights):
            raise ValueError(
                "The number of loss functions must match the number of weights.",
            )

        self.loss_functions = nn.ModuleList(loss_functions)
        self.weights = weights

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        """
        Compute the weighted sum of all specified losses.

        Parameters:

        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.

        Returns:
        - A dictionary with the total loss and individual losses.
        """
        total_loss = 0.0
        loss_details = {}

        for i, loss_fn in enumerate(self.loss_functions):
            loss_output = loss_fn(
                pred,
                target,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mask=mask,
                model=model,
            )

            # Assume each loss function returns a dictionary with a "loss" key
            individual_loss = loss_output.pop("loss")
            weighted_loss = self.weights[i] * individual_loss

            total_loss += weighted_loss

            # Add other keys from the loss output to the details
            for key, val in loss_output.items():
                loss_details[key] = val

        loss_details["loss"] = total_loss
        return loss_details


class LayeredWeightedPhysicsLoss(BaseLoss):
    def __init__(self, base_weight: float) -> None:
        super().__init__()
        self.base_weight = base_weight

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        total_loss = 0.0
        loss_details = {}

        layer_keys = sorted(model.layer_residuals.keys())
        L = len(layer_keys)

        # Compute raw weights (geometric decay)
        raw_weights = [self.base_weight ** (L - idx - 1) for idx in range(L)]

        # Normalize so weights sum to 1
        weight_sum = sum(raw_weights)
        norm_weights = [w / weight_sum for w in raw_weights]

        for key, weight in zip(layer_keys, norm_weights):
            residual = model.layer_residuals[key]
            total_loss = total_loss + weight * residual
            loss_details[f"layer_{key}_residual"] = residual.item()
            loss_details[f"layer_{key}_weight"] = weight

        loss_details["loss"] = total_loss
        loss_details["Layered Weighted Physics Loss"] = total_loss.item()
        return loss_details
