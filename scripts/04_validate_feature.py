import argparse
import logging

import pandas as pd
import torch
from tqdm.auto import tqdm

from gelu_1l.analysis import get_post_activations
from gelu_1l.artifacts import ensure_dir, save_csv, save_json
from gelu_1l.cli import add_common_args, configure_logging, feature_dir
from gelu_1l.constants import DEFAULT_FEATURE_ID, HOOK_POST
from gelu_1l.interventions import make_feature_ablation_hook, make_feature_boost_hook
from gelu_1l.loading import batched, load_gelu_1l, load_sae, load_texts, resolve_device, tokenize_batch
from gelu_1l.plotting import plot_validation_summary

logger = logging.getLogger(__name__)



@torch.no_grad()
def loss_per_prompt(model, tokens: torch.Tensor, hooks) -> list[float]:
    if hooks:
        loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            loss_per_token=True,
            fwd_hooks=hooks,
        )
    else:
        loss = model(tokens, return_type="loss", loss_per_token=True)
    return loss.mean(dim=-1).detach().float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Causally validate one GELU-1L SAE feature.")
    add_common_args(parser)
    parser.add_argument("--feature-id", type=int, default=DEFAULT_FEATURE_ID)
    parser.add_argument("--max-texts", type=int, default=20_000)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--n-prompts", type=int, default=512)
    parser.add_argument("--boost-factor", type=float, default=2.0)
    args = parser.parse_args()

    configure_logging(args.log_level)
    device = resolve_device(args.device)
    out_dir = ensure_dir(feature_dir(args.artifacts_dir, args.feature_id))

    model = load_gelu_1l(device=device, cache_dir=args.cache_dir)
    sae = load_sae(run=args.sae_run, device=device, cache_dir=args.cache_dir)
    texts = load_texts(max_texts=args.max_texts, cache_dir=args.cache_dir)

    prompt_rows = []
    with torch.no_grad():
        for batch_index, (text_offset, batch_texts) in enumerate(
            tqdm(batched(texts, args.batch_size), total=(len(texts) + args.batch_size - 1) // args.batch_size)
        ):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tokens = tokenize_batch(model, batch_texts, seq_len=args.seq_len, device=device)
            post = get_post_activations(model, tokens).reshape(-1, sae.act_size)
            feature_acts = sae.feature_activation(post, args.feature_id).reshape(tokens.shape[0], -1)
            max_values = feature_acts.max(dim=1).values.detach().float().cpu().tolist()
            for row_index, max_activation in enumerate(max_values):
                prompt_rows.append(
                    {
                        "text_id": text_offset + row_index,
                        "max_activation": max_activation,
                        "text": batch_texts[row_index],
                    }
                )

    prompt_scores = pd.DataFrame(prompt_rows)
    n_each = args.n_prompts // 2
    sorted_scores = prompt_scores.sort_values("max_activation", ascending=False)
    activating = sorted_scores.head(n_each).copy()
    activating["group"] = "activating"
    inactive = sorted_scores.tail(n_each).copy()
    inactive["group"] = "matched_low_activation"
    selected = pd.concat([activating, inactive], ignore_index=True)

    validation_rows = []
    ablation_hook = make_feature_ablation_hook(sae, args.feature_id)
    boost_hook = make_feature_boost_hook(sae, args.feature_id, boost_factor=args.boost_factor)
    for _, batch_rows in batched(selected.to_dict("records"), args.batch_size):
        batch_prompts = [row["text"] for row in batch_rows]
        tokens = tokenize_batch(model, batch_prompts, seq_len=args.seq_len, device=device)
        original = loss_per_prompt(model, tokens, hooks=[])
        ablated = loss_per_prompt(model, tokens, hooks=[(HOOK_POST, ablation_hook)])
        boosted = loss_per_prompt(model, tokens, hooks=[(HOOK_POST, boost_hook)])
        for row, original_loss, ablated_loss, boosted_loss in zip(batch_rows, original, ablated, boosted):
            validation_rows.append(
                {
                    "group": row["group"],
                    "text_id": int(row["text_id"]),
                    "max_activation": float(row["max_activation"]),
                    "original_loss": original_loss,
                    "ablated_loss": ablated_loss,
                    "boosted_loss": boosted_loss,
                    "ablation_delta_loss": ablated_loss - original_loss,
                    "boost_delta_loss": boosted_loss - original_loss,
                    "text_preview": row["text"][:240].replace("\n", "\\n"),
                }
            )

    validation = pd.DataFrame(validation_rows)
    summary = (
        validation.groupby("group", as_index=False)
        .agg(
            n_prompts=("text_id", "count"),
            mean_max_activation=("max_activation", "mean"),
            original_loss=("original_loss", "mean"),
            ablated_loss=("ablated_loss", "mean"),
            boosted_loss=("boosted_loss", "mean"),
            ablation_delta_loss=("ablation_delta_loss", "mean"),
            boost_delta_loss=("boost_delta_loss", "mean"),
        )
        .sort_values("group")
    )

    save_csv(out_dir / "prompt_scores.csv", prompt_scores.drop(columns=["text"]))
    save_csv(out_dir / "causal_validation.csv", validation)
    save_csv(out_dir / "causal_validation_summary.csv", summary)
    save_json(
        out_dir / "validation_config.json",
        {
            "feature_id": args.feature_id,
            "n_prompts": args.n_prompts,
            "boost_factor": args.boost_factor,
            "max_texts": args.max_texts,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "device": device,
        },
    )
    plot_validation_summary(summary, out_dir / "causal_validation.png", args.feature_id)

    logger.info("Wrote causal validation artifacts to %s", out_dir)
    logger.info("Validation summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
