import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from gelu_1l.constants import DEFAULT_MODEL_NAME, DEFAULT_SAE_REPO, HOOK_POST, MLP_POST_SIZE


def normalize_sae_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("model_name", DEFAULT_MODEL_NAME)
    cfg.setdefault("site", "post")
    cfg.setdefault("layer", 0)
    cfg.setdefault("act_name", HOOK_POST)
    cfg.setdefault("d_mlp", MLP_POST_SIZE)
    cfg.setdefault("act_size", cfg["d_mlp"])
    cfg.setdefault("dict_mult", 8)
    cfg.setdefault("dict_size", cfg["act_size"] * cfg["dict_mult"])
    cfg.setdefault("l1_coeff", 3e-4)
    cfg.setdefault("enc_dtype", "fp32")
    cfg.setdefault("seed", 0)
    return cfg


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder architecture used in Neel Nanda's GELU-1L SAE run."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = normalize_sae_cfg(cfg)
        dtype = torch.float32
        if self.cfg["enc_dtype"] == "fp16":
            dtype = torch.float16
        if self.cfg["enc_dtype"] == "bf16":
            dtype = torch.bfloat16

        act_size = int(self.cfg["act_size"])
        dict_size = int(self.cfg["dict_size"])
        torch.manual_seed(int(self.cfg["seed"]))

        self.W_enc = nn.Parameter(torch.empty(act_size, dict_size, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(dict_size, act_size, dtype=dtype))
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        self.b_enc = nn.Parameter(torch.zeros(dict_size, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(act_size, dtype=dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.d_hidden = dict_size
        self.l1_coeff = float(self.cfg["l1_coeff"])

    @classmethod
    def from_files(cls, cfg_path: Path, checkpoint_path: Path, device: str | torch.device) -> "SparseAutoencoder":
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        sae = cls(cfg).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        sae.load_state_dict(state_dict)
        sae.eval()
        return sae

    @classmethod
    def from_hf(
        cls,
        run: int | str,
        cache_dir: Path,
        device: str | torch.device,
        repo_id: str = DEFAULT_SAE_REPO,
    ) -> "SparseAutoencoder":
        version = {"run1": 25, "run2": 47}.get(run, run)
        cfg_path = Path(
            hf_hub_download(repo_id=repo_id, filename=f"{version}_cfg.json", cache_dir=str(cache_dir))
        )
        checkpoint_path = Path(
            hf_hub_download(repo_id=repo_id, filename=f"{version}.pt", cache_dir=str(cache_dir))
        )
        return cls.from_files(cfg_path=cfg_path, checkpoint_path=checkpoint_path, device=device)

    @property
    def act_size(self) -> int:
        return int(self.cfg["act_size"])

    @property
    def dict_size(self) -> int:
        return int(self.cfg["dict_size"])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_acts = self.encode(x)
        reconstruction = self.decode(feature_acts)
        l2_loss = (reconstruction.float() - x.float()).pow(2).sum(dim=-1).mean()
        l1_loss = self.l1_coeff * feature_acts.float().abs().sum()
        loss = l2_loss + l1_loss
        return loss, reconstruction, feature_acts, l2_loss, l1_loss

    def feature_activation(self, x: torch.Tensor, feature_id: int) -> torch.Tensor:
        direction = self.W_enc[:, feature_id]
        return F.relu((x - self.b_dec) @ direction + self.b_enc[feature_id])

    def feature_contribution(self, x: torch.Tensor, feature_id: int) -> torch.Tensor:
        feature_acts = self.feature_activation(x, feature_id)
        return feature_acts.unsqueeze(-1) * self.W_dec[feature_id]
