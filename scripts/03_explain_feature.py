import argparse
import logging

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from gelu_1l.analysis import (
    TopOccurrenceCollector,
    feature_logit_effects,
    get_post_activations,
    neuron_alignment,
    occurrence_row,
    top_feature_acts,
)
from gelu_1l.artifacts import ensure_dir, save_csv, save_json
from gelu_1l.cli import add_common_args, configure_logging, feature_dir
from gelu_1l.constants import DEFAULT_FEATURE_ID
from gelu_1l.loading import batched, load_gelu_1l, load_sae, load_texts, resolve_device, tokenize_batch
from gelu_1l.plotting import plot_activation_histogram

logger = logging.getLogger(__name__)

CONTEXT_COLUMNS = [
    "text_id",
    "position",
    "activation",
    "source_text_preview",
    "left_context",
    "token",
    "right_context",
    "window",
]
BIN_CONTEXT_COLUMNS = [*CONTEXT_COLUMNS, "bin_index", "bin_left", "bin_right"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain one GELU-1L SAE feature with examples and linear effects.")
    add_common_args(parser)
    parser.add_argument("--feature-id", type=int, default=DEFAULT_FEATURE_ID)
    parser.add_argument("--max-texts", type=int, default=20_000)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--examples-per-bin", type=int, default=3)
    parser.add_argument("--context-radius", type=int, default=8)
    args = parser.parse_args()

    configure_logging(args.log_level)
    device = resolve_device(args.device)
    out_dir = ensure_dir(feature_dir(args.artifacts_dir, args.feature_id))

    model = load_gelu_1l(device=device, cache_dir=args.cache_dir)
    sae = load_sae(run=args.sae_run, device=device, cache_dir=args.cache_dir)
    texts = load_texts(max_texts=args.max_texts, cache_dir=args.cache_dir)

    collector = TopOccurrenceCollector(limit=args.top_k)
    max_activation = 0.0
    positive_count = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_index, (text_offset, batch_texts) in enumerate(
            tqdm(batched(texts, args.batch_size), total=(len(texts) + args.batch_size - 1) // args.batch_size)
        ):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tokens = tokenize_batch(model, batch_texts, seq_len=args.seq_len, device=device)
            post = get_post_activations(model, tokens).reshape(-1, sae.act_size)
            feature_acts = sae.feature_activation(post, args.feature_id).reshape(tokens.shape[0], -1)
            max_activation = max(max_activation, float(feature_acts.max().item()))
            positive_count += int((feature_acts > 0).sum().item())
            total_tokens += int(feature_acts.numel())

            values, indices = top_feature_acts(feature_acts, k=args.top_k)
            for value, flat_index in zip(values, indices):
                if value <= 0:
                    continue
                batch_row = flat_index // feature_acts.shape[1]
                pos = flat_index % feature_acts.shape[1]
                collector.add(
                    float(value),
                    occurrence_row(
                        model,
                        tokens=tokens,
                        texts=batch_texts,
                        text_offset=text_offset,
                        batch_row=batch_row,
                        pos=pos,
                        activation=float(value),
                        radius=args.context_radius,
                    ),
                )

    top_contexts = pd.DataFrame(collector.rows(), columns=CONTEXT_COLUMNS)
    save_csv(out_dir / "top_contexts.csv", top_contexts)

    assert max_activation > 0, "Feature did not activate in the scanned texts"

    edges = np.linspace(0.0, max_activation, args.bins + 1)
    histogram_counts = np.zeros(args.bins, dtype=np.int64)
    examples: dict[int, list[dict[str, object]]] = {index: [] for index in range(args.bins)}

    with torch.no_grad():
        for batch_index, (text_offset, batch_texts) in enumerate(
            tqdm(batched(texts, args.batch_size), total=(len(texts) + args.batch_size - 1) // args.batch_size)
        ):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tokens = tokenize_batch(model, batch_texts, seq_len=args.seq_len, device=device)
            post = get_post_activations(model, tokens).reshape(-1, sae.act_size)
            feature_acts = sae.feature_activation(post, args.feature_id).reshape(tokens.shape[0], -1)
            flat = feature_acts.detach().flatten().float().cpu()
            active = flat[flat > 0]
            histogram_counts += np.histogram(active.numpy(), bins=edges)[0]

            for bin_index in range(args.bins):
                missing = args.examples_per_bin - len(examples[bin_index])
                if missing <= 0:
                    continue
                left, right = edges[bin_index], edges[bin_index + 1]
                candidate_indices = torch.nonzero((flat > left) & (flat <= right)).flatten()
                if candidate_indices.numel() == 0:
                    continue
                selected_values, local_order = torch.topk(
                    flat[candidate_indices],
                    k=min(missing, candidate_indices.numel()),
                )
                selected_indices = candidate_indices[local_order]
                for value, flat_index in zip(selected_values.tolist(), selected_indices.tolist()):
                    batch_row = flat_index // feature_acts.shape[1]
                    pos = flat_index % feature_acts.shape[1]
                    row = occurrence_row(
                        model,
                        tokens=tokens,
                        texts=batch_texts,
                        text_offset=text_offset,
                        batch_row=batch_row,
                        pos=pos,
                        activation=float(value),
                        radius=args.context_radius,
                    )
                    row["bin_index"] = bin_index
                    row["bin_left"] = float(left)
                    row["bin_right"] = float(right)
                    examples[bin_index].append(row)

    histogram = pd.DataFrame(
        {
            "bin_index": np.arange(args.bins),
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": histogram_counts,
        }
    )
    uniform_examples = pd.DataFrame(
        [row for bin_rows in examples.values() for row in bin_rows],
        columns=BIN_CONTEXT_COLUMNS,
    )
    save_csv(out_dir / "activation_histogram.csv", histogram)
    save_csv(out_dir / "uniform_bin_examples.csv", uniform_examples)
    plot_activation_histogram(histogram, out_dir / "activation_histogram.png", args.feature_id)

    logit_effects = feature_logit_effects(model, sae, feature_id=args.feature_id)
    alignment = neuron_alignment(sae, feature_id=args.feature_id)
    save_csv(out_dir / "logit_effects.csv", logit_effects)
    save_csv(out_dir / "neuron_alignment.csv", alignment)
    save_json(
        out_dir / "feature_summary.json",
        {
            "feature_id": args.feature_id,
            "sae_run": args.sae_run,
            "max_activation": max_activation,
            "positive_token_count": positive_count,
            "total_token_count": total_tokens,
            "activation_rate": positive_count / total_tokens,
            "top_contexts": len(top_contexts),
            "uniform_examples": len(uniform_examples),
        },
    )

    logger.info("Wrote feature explanation artifacts to %s", out_dir)
    logger.info("Top activating contexts:\n%s", top_contexts.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
