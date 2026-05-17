import torch

from gelu_1l.interventions import apply_feature_delta
from gelu_1l.sae import SparseAutoencoder


def test_apply_feature_delta_subtracts_feature_contribution() -> None:
    sae = SparseAutoencoder({"act_size": 2, "dict_size": 1, "seed": 1})
    with torch.no_grad():
        sae.W_enc.copy_(torch.tensor([[1.0], [0.0]]))
        sae.W_dec.copy_(torch.tensor([[2.0, 3.0]]))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    mlp_post = torch.tensor([[[4.0, 5.0]]])
    changed = apply_feature_delta(mlp_post, sae=sae, feature_id=0, multiplier=-1.0)
    assert torch.allclose(changed, torch.tensor([[[-4.0, -7.0]]]))
