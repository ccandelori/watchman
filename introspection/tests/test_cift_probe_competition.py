from __future__ import annotations

import unittest

from aegis_introspection.cift_probe_competition import (
    CiftProbeCompetitionConfig,
    CiftProbeCompetitionError,
    CiftProbeRun,
    cift_probe_competition_report_from_mapping,
    cift_probe_competition_report_to_json,
    compare_cift_probe_candidates,
    promotion_paper_method_from_probe_competition,
)


def _paper_probe(metric_value: float) -> CiftProbeRun:
    return CiftProbeRun(
        source_report_id="synthetic-paper-mlp-report",
        probe_architecture="mlp_128_64_1",
        training_loss="bce_with_l1_softplus_weight_sparsity",
        training_dataset_id="synthetic-cift-lab",
        training_dataset_sha256="a" * 64,
        task_name="safe_secret_vs_exfiltration",
        evaluation_split_id="synthetic-cift-lab/sealed-holdout",
        evaluation_split_manifest_id="synthetic-cift-lab/sealed-holdout/manifest",
        evaluation_split_sha256="b" * 64,
        metric_name="sealed_holdout_macro_f1",
        metric_value=metric_value,
        metric_confidence_interval_low=metric_value - 0.01,
        metric_confidence_interval_high=metric_value + 0.01,
        random_seeds=(11, 17, 23),
        hyperparameter_search_trials=8,
        operating_threshold=0.5,
        false_positive_rate=0.04,
        true_positive_rate=0.91,
    )


def _linear_probe(
    metric_value: float,
    evaluation_split_id: str,
    metric_confidence_interval_low: float | None = None,
    training_dataset_sha256: str = "a" * 64,
    hyperparameter_search_trials: int = 8,
    random_seeds: tuple[int, ...] = (11, 17, 23),
) -> CiftProbeRun:
    return CiftProbeRun(
        source_report_id="synthetic-linear-report",
        probe_architecture="linear_logistic_regression",
        training_loss="bce",
        training_dataset_id="synthetic-cift-lab",
        training_dataset_sha256=training_dataset_sha256,
        task_name="safe_secret_vs_exfiltration",
        evaluation_split_id=evaluation_split_id,
        evaluation_split_manifest_id="synthetic-cift-lab/sealed-holdout/manifest",
        evaluation_split_sha256="b" * 64,
        metric_name="sealed_holdout_macro_f1",
        metric_value=metric_value,
        metric_confidence_interval_low=(
            metric_value - 0.01 if metric_confidence_interval_low is None else metric_confidence_interval_low
        ),
        metric_confidence_interval_high=metric_value + 0.01,
        random_seeds=random_seeds,
        hyperparameter_search_trials=hyperparameter_search_trials,
        operating_threshold=0.5,
        false_positive_rate=0.03,
        true_positive_rate=0.93,
    )


def _competition_config(paper_metric: float, candidate_metric: float) -> CiftProbeCompetitionConfig:
    return CiftProbeCompetitionConfig(
        report_id="synthetic-linear-vs-mlp-report",
        paper_probe=_paper_probe(metric_value=paper_metric),
        candidate_probe=_linear_probe(
            metric_value=candidate_metric,
            evaluation_split_id="synthetic-cift-lab/sealed-holdout",
        ),
        higher_is_better=True,
        created_at="2026-06-24T00:00:00Z",
    )


class CiftProbeCompetitionTest(unittest.TestCase):
    def test_competition_allows_challenger_that_beats_paper_mlp(self) -> None:
        report = compare_cift_probe_candidates(_competition_config(paper_metric=0.9, candidate_metric=0.92))

        self.assertEqual("cift_probe_competition/v1", report.schema_version)
        self.assertEqual("linear_logistic_regression", report.winner_probe_architecture)
        self.assertTrue(report.candidate_meets_or_exceeds_paper)
        self.assertAlmostEqual(0.02, report.candidate_delta, places=12)
        self.assertEqual("a" * 64, report.training_dataset_sha256)
        self.assertEqual("synthetic-cift-lab/sealed-holdout/manifest", report.evaluation_split_manifest_id)
        self.assertEqual("b" * 64, report.evaluation_split_sha256)
        self.assertEqual((11, 17, 23), report.random_seeds)

    def test_competition_rejects_mismatched_evaluation_split(self) -> None:
        config = CiftProbeCompetitionConfig(
            report_id="synthetic-linear-vs-mlp-report",
            paper_probe=_paper_probe(metric_value=0.9),
            candidate_probe=_linear_probe(
                metric_value=0.92,
                evaluation_split_id="synthetic-cift-lab/heldout",
            ),
            higher_is_better=True,
            created_at="2026-06-24T00:00:00Z",
        )

        with self.assertRaisesRegex(CiftProbeCompetitionError, "evaluation_split_id"):
            compare_cift_probe_candidates(config)

    def test_competition_rejects_mismatched_dataset_hash(self) -> None:
        config = CiftProbeCompetitionConfig(
            report_id="synthetic-linear-vs-mlp-report",
            paper_probe=_paper_probe(metric_value=0.9),
            candidate_probe=_linear_probe(
                metric_value=0.92,
                evaluation_split_id="synthetic-cift-lab/sealed-holdout",
                training_dataset_sha256="c" * 64,
            ),
            higher_is_better=True,
            created_at="2026-06-24T00:00:00Z",
        )

        with self.assertRaisesRegex(CiftProbeCompetitionError, "training_dataset_sha256"):
            compare_cift_probe_candidates(config)

    def test_competition_rejects_candidate_with_larger_search_budget(self) -> None:
        config = CiftProbeCompetitionConfig(
            report_id="synthetic-linear-vs-mlp-report",
            paper_probe=_paper_probe(metric_value=0.9),
            candidate_probe=_linear_probe(
                metric_value=0.92,
                evaluation_split_id="synthetic-cift-lab/sealed-holdout",
                hyperparameter_search_trials=16,
            ),
            higher_is_better=True,
            created_at="2026-06-24T00:00:00Z",
        )

        with self.assertRaisesRegex(CiftProbeCompetitionError, "hyperparameter_search_trials"):
            compare_cift_probe_candidates(config)

    def test_competition_requires_repeated_seed_evidence(self) -> None:
        config = CiftProbeCompetitionConfig(
            report_id="synthetic-linear-vs-mlp-report",
            paper_probe=_paper_probe(metric_value=0.9),
            candidate_probe=_linear_probe(
                metric_value=0.92,
                evaluation_split_id="synthetic-cift-lab/sealed-holdout",
                random_seeds=(11,),
            ),
            higher_is_better=True,
            created_at="2026-06-24T00:00:00Z",
        )

        with self.assertRaisesRegex(CiftProbeCompetitionError, "random_seeds"):
            compare_cift_probe_candidates(config)

    def test_competition_rejects_point_win_without_confidence_support_for_promotion(self) -> None:
        report = compare_cift_probe_candidates(
            CiftProbeCompetitionConfig(
                report_id="synthetic-linear-vs-mlp-report",
                paper_probe=_paper_probe(metric_value=0.9),
                candidate_probe=_linear_probe(
                    metric_value=0.92,
                    metric_confidence_interval_low=0.89,
                    evaluation_split_id="synthetic-cift-lab/sealed-holdout",
                ),
                higher_is_better=True,
                created_at="2026-06-24T00:00:00Z",
            )
        )

        self.assertFalse(report.candidate_meets_or_exceeds_paper)
        with self.assertRaisesRegex(CiftProbeCompetitionError, "confidence interval"):
            promotion_paper_method_from_probe_competition(report)

    def test_competition_rejects_non_paper_reference_probe(self) -> None:
        config = CiftProbeCompetitionConfig(
            report_id="synthetic-linear-vs-mlp-report",
            paper_probe=_linear_probe(
                metric_value=0.9,
                evaluation_split_id="synthetic-cift-lab/sealed-holdout",
            ),
            candidate_probe=_linear_probe(
                metric_value=0.92,
                evaluation_split_id="synthetic-cift-lab/sealed-holdout",
            ),
            higher_is_better=True,
            created_at="2026-06-24T00:00:00Z",
        )

        with self.assertRaisesRegex(CiftProbeCompetitionError, "paper_probe"):
            compare_cift_probe_candidates(config)

    def test_competition_report_round_trips_through_json(self) -> None:
        report = compare_cift_probe_candidates(_competition_config(paper_metric=0.9, candidate_metric=0.92))

        report_json = cift_probe_competition_report_to_json(report)
        candidate_probe = report_json["candidate_probe"]
        paper_probe = report_json["paper_probe"]
        if not isinstance(candidate_probe, dict) or not isinstance(paper_probe, dict):
            raise AssertionError("probe records must be objects.")
        self.assertAlmostEqual(0.07, candidate_probe["false_negative_rate"], places=12)
        self.assertAlmostEqual(0.09, paper_probe["false_negative_rate"], places=12)

        parsed = cift_probe_competition_report_from_mapping(report_json)

        self.assertEqual(report, parsed)

    def test_competition_report_rejects_inconsistent_false_negative_rate(self) -> None:
        report_json = cift_probe_competition_report_to_json(
            compare_cift_probe_candidates(_competition_config(paper_metric=0.9, candidate_metric=0.92))
        )
        candidate_probe = report_json["candidate_probe"]
        if not isinstance(candidate_probe, dict):
            raise AssertionError("candidate_probe must be an object.")
        candidate_probe["false_negative_rate"] = 0.25

        with self.assertRaisesRegex(CiftProbeCompetitionError, "false_negative_rate"):
            cift_probe_competition_report_from_mapping(report_json)

    def test_promotion_method_uses_challenger_only_when_it_meets_paper(self) -> None:
        report = compare_cift_probe_candidates(_competition_config(paper_metric=0.9, candidate_metric=0.92))

        method = promotion_paper_method_from_probe_competition(report)

        self.assertEqual("linear_logistic_regression", method.probe_architecture)
        self.assertEqual("bce", method.training_loss)
        self.assertEqual("synthetic-linear-vs-mlp-report", method.head_to_head_report_id)
        self.assertEqual(0.9, method.paper_probe_metric_value)
        self.assertEqual(0.92, method.candidate_probe_metric_value)

    def test_promotion_method_rejects_challenger_that_underperforms_paper(self) -> None:
        report = compare_cift_probe_candidates(_competition_config(paper_metric=0.92, candidate_metric=0.9))

        with self.assertRaisesRegex(CiftProbeCompetitionError, "candidate metric"):
            promotion_paper_method_from_probe_competition(report)


if __name__ == "__main__":
    unittest.main()
