import tempfile
import unittest
from pathlib import Path

from aegis_introspection.probe_report_summary import (
    ProbeReportSummaryError,
    parse_feature_key,
    parse_probe_report_summary,
    render_probe_report_markdown,
    sorted_features_by_score,
    write_probe_report_markdown,
)


def _probe_report() -> dict[str, object]:
    return {
        "source_model_id": "Qwen/Qwen3-0.6B",
        "source_revision": "main",
        "source_selected_device": "cpu",
        "label_names": ["benign", "exfiltration_intent", "secret_present_safe"],
        "fold_count": 5,
        "random_seed": 42,
        "regularization_c": 1.0,
        "max_iter": 1000,
        "best_feature_key": "mean_pool_layer_14",
        "features": [
            {
                "feature_key": "final_token_layer_00",
                "example_count": 90,
                "feature_count": 1024,
                "accuracy_mean": 0.3,
                "accuracy_std": 0.01,
                "macro_f1_mean": 0.2,
                "macro_f1_std": 0.01,
                "confusion_matrix": [[1, 0], [0, 1]],
                "folds": [],
            },
            {
                "feature_key": "mean_pool_layer_14",
                "example_count": 90,
                "feature_count": 1024,
                "accuracy_mean": 0.95,
                "accuracy_std": 0.02,
                "macro_f1_mean": 0.94,
                "macro_f1_std": 0.03,
                "confusion_matrix": [[1, 0], [0, 1]],
                "folds": [],
            },
        ],
    }


class ProbeReportSummaryTest(unittest.TestCase):
    def test_parse_feature_key_returns_pooling_and_layer(self) -> None:
        pooling_method, layer_index = parse_feature_key("mean_pool_layer_14")

        self.assertEqual("mean_pool", pooling_method)
        self.assertEqual(14, layer_index)

    def test_parse_feature_key_rejects_unknown_format(self) -> None:
        with self.assertRaises(ProbeReportSummaryError):
            parse_feature_key("mean_pool_14")

    def test_sorted_features_by_score_orders_highest_macro_f1_first(self) -> None:
        summary = parse_probe_report_summary(_probe_report())

        sorted_features = sorted_features_by_score(summary)

        self.assertEqual("mean_pool_layer_14", sorted_features[0].feature_key)
        self.assertEqual("final_token_layer_00", sorted_features[1].feature_key)

    def test_render_probe_report_markdown_includes_ranking(self) -> None:
        summary = parse_probe_report_summary(_probe_report())

        markdown = render_probe_report_markdown(summary)

        self.assertIn("# Probe Layer Sweep Summary", markdown)
        self.assertIn("`mean_pool_layer_14`", markdown)
        self.assertIn("| Pooling | Best Layer | Best Macro F1 | Best Accuracy |", markdown)

    def test_write_probe_report_markdown_creates_file(self) -> None:
        summary = parse_probe_report_summary(_probe_report())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.md"
            write_probe_report_markdown(output_path, summary)
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Probe Layer Sweep Summary", text)


if __name__ == "__main__":
    unittest.main()
