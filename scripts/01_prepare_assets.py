import argparse
import importlib.metadata
import logging

from gelu_1l.artifacts import ensure_dir, save_json
from gelu_1l.cli import add_common_args, configure_logging
from gelu_1l.constants import DEFAULT_DATASET_NAME, DEFAULT_MODEL_NAME, DEFAULT_SAE_REPO
from gelu_1l.loading import load_gelu_1l, load_sae, load_texts, resolve_device

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify GELU-1L case-study assets.")
    add_common_args(parser)
    args = parser.parse_args()

    configure_logging(args.log_level)
    device = resolve_device(args.device)
    ensure_dir(args.artifacts_dir)
    ensure_dir(args.cache_dir)

    model = load_gelu_1l(device=device, cache_dir=args.cache_dir)
    sae = load_sae(run=args.sae_run, device=device, cache_dir=args.cache_dir)
    texts = load_texts(max_texts=8, cache_dir=args.cache_dir)

    model_info = {
        "model_name": DEFAULT_MODEL_NAME,
        "device": device,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "d_mlp": model.cfg.d_mlp,
        "n_heads": model.cfg.n_heads,
        "d_vocab": model.cfg.d_vocab,
        "n_ctx": model.cfg.n_ctx,
    }
    sae_info = {
        "repo_id": DEFAULT_SAE_REPO,
        "run": args.sae_run,
        "act_name": sae.cfg["act_name"],
        "act_size": sae.act_size,
        "dict_size": sae.dict_size,
        "l1_coeff": sae.l1_coeff,
    }
    dataset_info = {
        "dataset_name": DEFAULT_DATASET_NAME,
        "sample_texts_loaded": len(texts),
    }
    packages = {
        "torch": importlib.metadata.version("torch"),
        "transformer-lens": importlib.metadata.version("transformer-lens"),
        "datasets": importlib.metadata.version("datasets"),
    }

    save_json(
        args.artifacts_dir / "assets.json",
        {
            "model": model_info,
            "sae": sae_info,
            "dataset": dataset_info,
            "packages": packages,
        },
    )

    logger.info("Prepared GELU-1L SAE case-study assets")
    logger.info("Model: %s on %s", DEFAULT_MODEL_NAME, device)
    logger.info(
        "Model shape: %s layer, d_model=%s, d_mlp=%s",
        model.cfg.n_layers,
        model.cfg.d_model,
        model.cfg.d_mlp,
    )
    logger.info("SAE: %s run %s, dict_size=%s", DEFAULT_SAE_REPO, args.sae_run, sae.dict_size)
    logger.info("Dataset: %s, verified %s rows", DEFAULT_DATASET_NAME, len(texts))
    logger.info("Wrote %s", args.artifacts_dir / "assets.json")


if __name__ == "__main__":
    main()
