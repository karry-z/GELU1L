from pathlib import Path

DEFAULT_MODEL_NAME = "gelu-1l"
DEFAULT_DATASET_NAME = "NeelNanda/c4-code-20k"
DEFAULT_SAE_REPO = "NeelNanda/sparse_autoencoder"
DEFAULT_SAE_RUN = 25
DEFAULT_FEATURE_ID = 8
DEFAULT_SEQ_LEN = 128
DEFAULT_BATCH_SIZE = 4
DEFAULT_ARTIFACTS_DIR = Path("artifacts/feature_case_study")
DEFAULT_CACHE_DIR = Path(".cache/gelu_1l")

HOOK_POST = "blocks.0.mlp.hook_post"
MLP_POST_SIZE = 2048
SAE_DICT_MULT = 8
SAE_DICT_SIZE = MLP_POST_SIZE * SAE_DICT_MULT

RESEARCH_SOURCES = {
    "TransformerLens docs": "https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html",
    "Neuroscope": "https://neuroscope.io/",
    "1L-Sparse-Autoencoder": "https://github.com/neelnanda-io/1L-Sparse-Autoencoder",
    "Towards Monosemanticity": "https://transformer-circuits.pub/2023/monosemantic-features/",
    "Gated SAE NeurIPS paper": "https://proceedings.neurips.cc/paper_files/paper/2024/file/01772a8b0420baec00c4d59fe2fbace6-Paper-Conference.pdf",
    "MLP linearization case studies": "https://www.lesswrong.com/posts/93nKtsDL6YY5fRbQv/case-studies-in-reverse-engineering-sparse-autoencoder",
}
