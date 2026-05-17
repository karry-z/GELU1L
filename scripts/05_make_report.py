import argparse
import logging

import pandas as pd

from gelu_1l.artifacts import ensure_dir, load_csv, load_json
from gelu_1l.cli import add_common_args, configure_logging, feature_dir
from gelu_1l.constants import DEFAULT_FEATURE_ID, RESEARCH_SOURCES

logger = logging.getLogger(__name__)


def markdown_table(frame: pd.DataFrame, columns: list[str], n: int) -> str:
    subset = frame[columns].head(n).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in subset.astype(str).to_dict("records"):
        cells = [row[column].replace("\n", "\\n").replace("|", "\\|") for column in columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a Markdown report for one GELU-1L SAE feature.")
    add_common_args(parser)
    parser.add_argument("--feature-id", type=int, default=DEFAULT_FEATURE_ID)
    args = parser.parse_args()

    configure_logging(args.log_level)
    out_dir = ensure_dir(feature_dir(args.artifacts_dir, args.feature_id))
    summary = load_json(out_dir / "feature_summary.json")
    top_contexts = load_csv(out_dir / "top_contexts.csv")
    uniform_examples = load_csv(out_dir / "uniform_bin_examples.csv")
    logit_effects = load_csv(out_dir / "logit_effects.csv")
    alignment = load_csv(out_dir / "neuron_alignment.csv")
    validation_summary = load_csv(out_dir / "causal_validation_summary.csv")

    context_tokens = ", ".join(token.replace("\n", "\\n") for token in top_contexts["token"].astype(str).tolist()[:8])
    positive_tokens = ", ".join(
        token.replace("\n", "\\n")
        for token in logit_effects.loc[logit_effects["direction"] == "positive", "token"].astype(str).tolist()[:8]
    )
    hypothesis = (
        f"Feature {args.feature_id} fires on contexts resembling its top activating token windows, "
        f"especially around tokens like {context_tokens}; its linearized decoder direction most promotes "
        f"next-token logits like {positive_tokens}."
    )

    activating = validation_summary.loc[validation_summary["group"] == "activating"]
    low = validation_summary.loc[validation_summary["group"] == "matched_low_activation"]
    active_delta = float(activating["ablation_delta_loss"].iloc[0])
    low_delta = float(low["ablation_delta_loss"].iloc[0])
    if active_delta > low_delta:
        causal_result = (
            f"Ablation raises mean loss more on activating prompts ({active_delta:.4f}) than on low-activation "
            f"prompts ({low_delta:.4f}), which supports the feature being causally relevant on its own examples."
        )
    else:
        causal_result = (
            f"Ablation does not selectively raise activating-prompt loss: activating delta={active_delta:.4f}, "
            f"low-activation delta={low_delta:.4f}. Treat the label as descriptive evidence, not a validated circuit claim."
        )
    sources = "\n".join(f"- [{name}]({url})" for name, url in RESEARCH_SOURCES.items())

    report = f"""# GELU-1L SAE Feature {args.feature_id} Case Study

## Conclusion
{hypothesis}

## Evidence
- SAE run: `{args.sae_run}`
- Hook site: post-GELU MLP activations, `blocks.0.mlp.hook_post`
- Max activation: `{summary["max_activation"]:.6f}`
- Activation rate: `{summary["activation_rate"]:.6%}` over `{summary["total_token_count"]}` scanned tokens
- Top-context artifact: `{out_dir / "top_contexts.csv"}`
- Histogram: `{out_dir / "activation_histogram.png"}`

### Top Activating Contexts
{markdown_table(top_contexts, ["activation", "text_id", "position", "window"], 12)}

### Uniform Activation-Bin Examples
{markdown_table(uniform_examples, ["bin_index", "activation", "text_id", "position", "window"], 12)}

### Direct Logit Effects
{markdown_table(logit_effects, ["direction", "rank", "token", "logit_effect"], 20)}

### Neuron Alignment
{markdown_table(alignment, ["rank", "neuron_id", "decoder_weight", "abs_decoder_weight", "sign"], 15)}

## Causal Validation
{causal_result}

{markdown_table(validation_summary, ["group", "n_prompts", "mean_max_activation", "ablation_delta_loss", "boost_delta_loss"], 4)}

Validation plot: `{out_dir / "causal_validation.png"}`

## Caveat
This is a mechanistic-interpretability case study, not a production classifier. The feature label is a hypothesis grounded in examples, decoder/logit effects, and intervention deltas. A weak or non-selective intervention result means the label should be treated as descriptive rather than causal.

## Sources
{sources}
"""
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
