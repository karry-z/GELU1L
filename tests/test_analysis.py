from types import SimpleNamespace

import torch

from gelu_1l.analysis import (
    FeatureStatsTracker,
    TopOccurrenceCollector,
    feature_logit_effects,
    format_token_window,
    neuron_alignment,
)
from gelu_1l.sae import SparseAutoencoder


def test_feature_stats_tracker_counts_rates_and_maxima() -> None:
    tracker = FeatureStatsTracker(dict_size=3)
    tracker.update(torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]]))
    frame = tracker.to_frame().sort_values("feature_id")
    assert frame["activation_count"].tolist() == [1, 1, 1]
    assert frame["total_tokens"].tolist() == [2, 2, 2]
    assert frame["max_activation"].tolist() == [1.0, 2.0, 3.0]


def test_top_occurrence_collector_handles_equal_activations() -> None:
    collector = TopOccurrenceCollector(limit=2)
    collector.add(1.0, {"id": "a"})
    collector.add(1.0, {"id": "b"})
    collector.add(2.0, {"id": "c"})
    rows = collector.rows()
    assert [row["id"] for row in rows] == ["c", "b"]


def test_format_token_window_marks_target_token() -> None:
    window = format_token_window(["a", "b", "c", "d"], pos=2, radius=1)
    assert window["left_context"] == "b"
    assert window["token"] == "c"
    assert window["right_context"] == "d"
    assert window["window"] == "b<<c>>d"


def test_neuron_alignment_returns_largest_decoder_weights() -> None:
    sae = SparseAutoencoder({"act_size": 4, "dict_size": 2, "seed": 1})
    with torch.no_grad():
        sae.W_dec[1].copy_(torch.tensor([0.1, -0.9, 0.3, 0.7]))
    frame = neuron_alignment(sae, feature_id=1, top_k=2)
    assert frame["neuron_id"].tolist() == [1, 3]
    assert frame["sign"].tolist() == ["negative", "positive"]


def test_feature_logit_effects_shape() -> None:
    sae = SparseAutoencoder({"act_size": 4, "dict_size": 2, "seed": 1})
    model = SimpleNamespace(
        W_U=torch.randn(3, 7),
        blocks=[SimpleNamespace(mlp=SimpleNamespace(W_out=torch.randn(4, 3)))],
        tokenizer=SimpleNamespace(decode=lambda ids: f"tok_{ids[0]}"),
    )
    frame = feature_logit_effects(model, sae, feature_id=1, top_k=3)
    assert len(frame) == 6
    assert set(frame["direction"]) == {"positive", "negative"}
