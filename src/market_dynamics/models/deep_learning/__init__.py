"""Native deep sequence models with a common scalar-output interface."""

from market_dynamics.models.deep_learning.asset_embeddings import (
    AssetAgnosticModel,
    AssetConditionedModel,
    FixedPriorResidualModel,
    ZeroChannelSequenceModel,
)
from market_dynamics.models.deep_learning.factory import build_deep_model
from market_dynamics.models.deep_learning.family_adaptive_transformer import (
    FamilyAdaptiveTransformer,
)

__all__ = [
    "AssetAgnosticModel",
    "AssetConditionedModel",
    "FixedPriorResidualModel",
    "ZeroChannelSequenceModel",
    "FamilyAdaptiveTransformer",
    "build_deep_model",
]
