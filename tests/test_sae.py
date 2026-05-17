import torch

from gelu_1l.sae import SparseAutoencoder, normalize_sae_cfg


def test_normalize_sae_cfg_infers_shapes() -> None:
    cfg = normalize_sae_cfg({"d_mlp": 4, "dict_mult": 3})
    assert cfg["act_size"] == 4
    assert cfg["dict_size"] == 12
    assert cfg["act_name"] == "blocks.0.mlp.hook_post"


def test_sparse_autoencoder_encode_decode_shapes() -> None:
    sae = SparseAutoencoder({"act_size": 4, "dict_size": 6, "seed": 1})
    x = torch.randn(3, 4)
    feature_acts = sae.encode(x)
    reconstruction = sae.decode(feature_acts)
    assert feature_acts.shape == (3, 6)
    assert reconstruction.shape == (3, 4)


def test_feature_contribution_matches_feature_activation_times_decoder_row() -> None:
    sae = SparseAutoencoder({"act_size": 2, "dict_size": 3, "seed": 1})
    with torch.no_grad():
        sae.W_enc.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.5]]))
        sae.W_dec.copy_(torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    x = torch.tensor([[2.0, 1.0]])
    contribution = sae.feature_contribution(x, feature_id=0)
    assert torch.allclose(contribution, torch.tensor([[4.0, 6.0]]))
