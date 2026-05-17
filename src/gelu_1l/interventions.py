import torch


def apply_feature_delta(mlp_post: torch.Tensor, sae, feature_id: int, multiplier: float) -> torch.Tensor:
    flat = mlp_post.reshape(-1, mlp_post.shape[-1])
    contribution = sae.feature_contribution(flat, feature_id)
    changed = flat + multiplier * contribution
    return changed.reshape_as(mlp_post)


def make_feature_ablation_hook(sae, feature_id: int):
    def hook(mlp_post: torch.Tensor, hook):
        return apply_feature_delta(mlp_post, sae=sae, feature_id=feature_id, multiplier=-1.0)

    return hook


def make_feature_boost_hook(sae, feature_id: int, boost_factor: float):
    def hook(mlp_post: torch.Tensor, hook):
        return apply_feature_delta(
            mlp_post,
            sae=sae,
            feature_id=feature_id,
            multiplier=boost_factor - 1.0,
        )

    return hook
