from __future__ import annotations

import unittest

import numpy as np
from aegis_introspection.cift_paper_mlp import (
    CiftPaperMlpClassifier,
    CiftPaperMlpConfig,
    CiftPaperMlpError,
    paper_mlp_parameter_count,
)


class CiftPaperMlpTest(unittest.TestCase):
    def test_parameter_count_matches_paper_for_eight_layer_features(self) -> None:
        self.assertEqual(9473, paper_mlp_parameter_count(input_dim=8, hidden_layer_sizes=(128, 64)))

    def test_classifier_learns_simple_boundary_and_exposes_nonnegative_layer_weights(self) -> None:
        matrix = np.asarray(
            (
                (0.0, 0.0),
                (0.0, 0.1),
                (0.1, 0.0),
                (0.2, 0.1),
                (2.0, 2.0),
                (2.0, 2.2),
                (2.2, 2.0),
                (2.3, 2.1),
            ),
            dtype=np.float32,
        )
        labels = np.asarray((0, 0, 0, 0, 1, 1, 1, 1), dtype=np.int64)
        classifier = CiftPaperMlpClassifier(
            CiftPaperMlpConfig(
                input_dim=2,
                hidden_layer_sizes=(128, 64),
                learning_rate=0.05,
                max_epochs=220,
                batch_size=4,
                l1_softplus_weight=0.0001,
                random_seed=7,
            )
        )

        fitted = classifier.fit(matrix, labels)
        probabilities = fitted.predict_proba(np.asarray(((0.0, 0.0), (2.4, 2.1)), dtype=np.float32))
        layer_weights = fitted.softplus_layer_weights()

        self.assertIs(fitted, classifier)
        self.assertEqual((2, 2), probabilities.shape)
        self.assertEqual((0, 1), tuple(int(value) for value in fitted.classes_.tolist()))
        self.assertLess(probabilities[0, 1], 0.5)
        self.assertGreater(probabilities[1, 1], 0.5)
        self.assertTrue(bool(np.all(layer_weights > 0.0)))

    def test_classifier_rejects_feature_width_mismatch(self) -> None:
        classifier = CiftPaperMlpClassifier(
            CiftPaperMlpConfig(
                input_dim=2,
                hidden_layer_sizes=(128, 64),
                learning_rate=0.05,
                max_epochs=10,
                batch_size=4,
                l1_softplus_weight=0.0001,
                random_seed=7,
            )
        )
        matrix = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=np.float32)
        labels = np.asarray((0, 1), dtype=np.int64)

        with self.assertRaisesRegex(CiftPaperMlpError, "input_dim"):
            classifier.fit(matrix, labels)


if __name__ == "__main__":
    unittest.main()
