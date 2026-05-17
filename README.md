# GELU-1L SAE Feature Case Study

This repo implements a complete mechanism-interpretability case study for GELU-1L. It loads the TransformerLens `gelu-1l` model, Neel Nanda's pretrained sparse autoencoder for post-GELU MLP activations, scans real C4/code text, explains one SAE feature, and tests the feature with causal interventions.

Assumption: this is research tooling, not a product classifier. The report gives a feature hypothesis plus evidence; it does not claim the hypothesis is true unless the intervention result supports it.

## Why GELU-1L

GELU-1L is used in mechanism interpretability because it is small enough for circuit-level inspection while still trained on real text/code. This workflow follows the same kind of evidence used in SAE feature research: top activating examples, activation distributions, decoder/logit effects, neuron alignment, and feature-direction interventions.

Sources:

- [TransformerLens docs](https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html)
- [Neuroscope](https://neuroscope.io/)
- [1L-Sparse-Autoencoder](https://github.com/neelnanda-io/1L-Sparse-Autoencoder)
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/)
- [Gated SAE NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/01772a8b0420baec00c4d59fe2fbace6-Paper-Conference.pdf)
- [MLP linearization case studies](https://www.lesswrong.com/posts/93nKtsDL6YY5fRbQv/case-studies-in-reverse-engineering-sparse-autoencoder)

## Full Run

```bash
uv run python scripts/01_prepare_assets.py --device auto
uv run python scripts/02_find_features.py --max-texts 20000
uv run python scripts/03_explain_feature.py --feature-id 8
uv run python scripts/04_validate_feature.py --feature-id 8 --n-prompts 512
uv run python scripts/05_make_report.py --feature-id 8
```

Main output:

```text
artifacts/feature_case_study/feature_8/report.md
```

Supporting outputs include feature stats, top activating contexts, uniform activation-bin examples, direct logit effects, neuron alignment, causal validation CSVs, and plots.

## Fast Smoke Run

Use this before a full scan:

```bash
uv run python scripts/01_prepare_assets.py --device cpu
uv run python scripts/02_find_features.py --device cpu --max-texts 8 --max-batches 2 --batch-size 2
uv run python scripts/03_explain_feature.py --device cpu --feature-id 8 --max-texts 8 --max-batches 2 --batch-size 2
uv run python scripts/04_validate_feature.py --device cpu --feature-id 8 --max-texts 8 --max-batches 2 --n-prompts 4 --batch-size 2
uv run python scripts/05_make_report.py --feature-id 8
```

If feature `8` is inactive in the tiny smoke sample, use a feature from `artifacts/feature_case_study/top_candidate_features.csv` to exercise the non-empty explanation path.

## Tests

```bash
uv run pytest
```

The unit tests cover SAE shape inference, token-window extraction, feature statistics, neuron alignment, feature intervention math, and artifact JSON output. The smoke run above verifies the real model/data/script path.

## Interpretation Caveats

- The default SAE is pretrained checkpoint `25` from `NeelNanda/sparse_autoencoder`; the repo does not train a new SAE.
- The data is real `NeelNanda/c4-code-20k` text/code, but feature labels are still hypotheses.
- Logit effects are linearized from the SAE decoder direction through the MLP output and unembedding.
- Ablation/boosting at `blocks.0.mlp.hook_post` tests causal relevance for selected prompts, not global model behavior.
