import argparse
import logging

import torch
from tqdm.auto import tqdm

from gelu_1l.analysis import FeatureStatsTracker, get_post_activations
from gelu_1l.artifacts import ensure_dir, save_csv, save_json
from gelu_1l.cli import add_common_args, configure_logging
from gelu_1l.loading import batched, load_gelu_1l, load_sae, load_texts, resolve_device, tokenize_batch

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank GELU-1L SAE features on real C4/code text.")
    add_common_args(parser)
    parser.add_argument("--max-texts", type=int, default=20_000)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    configure_logging(args.log_level)
    device = resolve_device(args.device)
    ensure_dir(args.artifacts_dir)

    model = load_gelu_1l(device=device, cache_dir=args.cache_dir)
    sae = load_sae(run=args.sae_run, device=device, cache_dir=args.cache_dir)
    texts = load_texts(max_texts=args.max_texts, cache_dir=args.cache_dir)
    tracker = FeatureStatsTracker(dict_size=sae.dict_size)

    with torch.no_grad():
        for batch_index, (text_offset, batch_texts) in enumerate(
            tqdm(batched(texts, args.batch_size), total=(len(texts) + args.batch_size - 1) // args.batch_size)
        ):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tokens = tokenize_batch(model, batch_texts, seq_len=args.seq_len, device=device)
            post = get_post_activations(model, tokens).reshape(-1, sae.act_size)
            feature_acts = sae.encode(post)
            tracker.update(feature_acts)

    stats = tracker.to_frame()
    ranked = stats.head(args.top_n)
    save_csv(args.artifacts_dir / "feature_stats.csv", stats)
    save_csv(args.artifacts_dir / "top_candidate_features.csv", ranked)
    save_json(
        args.artifacts_dir / "find_features_config.json",
        {
            "max_texts": args.max_texts,
            "processed_tokens": int(stats["total_tokens"].iloc[0]),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "sae_run": args.sae_run,
            "device": device,
        },
    )

    logger.info("Wrote %s", args.artifacts_dir / "feature_stats.csv")
    logger.info("Top candidate features:\n%s", ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
