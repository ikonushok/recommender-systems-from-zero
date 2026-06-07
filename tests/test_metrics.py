import math
import unittest

import pandas as pd

from recsys_basics.metrics import (
    average_precision_at_k,
    dcg_at_k,
    evaluate_ranking_metrics,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class RankingMetricsTest(unittest.TestCase):
    def test_tiny_fixture_has_expected_top_k_metrics(self) -> None:
        relevant = ["A", "B"]
        recommended = ["A", "C", "B"]

        self.assertAlmostEqual(precision_at_k(relevant, recommended, k=3), 2 / 3)
        self.assertAlmostEqual(recall_at_k(relevant, recommended, k=3), 1.0)
        self.assertAlmostEqual(hit_rate_at_k(relevant, recommended, k=3), 1.0)
        self.assertAlmostEqual(average_precision_at_k(relevant, recommended, k=3), 5 / 6)

        expected_dcg = 1.0 + 1.0 / math.log2(4)
        expected_idcg = 1.0 + 1.0 / math.log2(3)
        self.assertAlmostEqual(dcg_at_k(relevant, recommended, k=3), expected_dcg)
        self.assertAlmostEqual(
            ndcg_at_k(relevant, recommended, k=3),
            expected_dcg / expected_idcg,
        )

    def test_empty_inputs_and_non_positive_k_return_zero(self) -> None:
        self.assertEqual(precision_at_k([], [], k=3), 0.0)
        self.assertEqual(recall_at_k([], ["A"], k=3), 0.0)
        self.assertEqual(hit_rate_at_k(["A"], [], k=3), 0.0)
        self.assertEqual(average_precision_at_k([], ["A"], k=3), 0.0)
        self.assertEqual(ndcg_at_k([], ["A"], k=3), 0.0)
        self.assertEqual(precision_at_k(["A"], ["A"], k=0), 0.0)

    def test_evaluation_groups_by_user_and_orders_by_rank(self) -> None:
        test_interactions = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "item_id": ["A", "B", "D"],
            }
        )
        recommendations = pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "item_id": ["B", "A", "C"],
                "rank": [3, 1, 2],
            }
        )

        result = evaluate_ranking_metrics(test_interactions, recommendations, k=3)
        by_user = result.set_index("user_id")

        self.assertAlmostEqual(by_user.loc[1, "precision@3"], 2 / 3)
        self.assertAlmostEqual(by_user.loc[1, "recall@3"], 1.0)
        self.assertAlmostEqual(by_user.loc[1, "hit_rate@3"], 1.0)
        self.assertAlmostEqual(by_user.loc[1, "map@3"], 5 / 6)
        self.assertAlmostEqual(by_user.loc[2, "precision@3"], 0.0)
        self.assertAlmostEqual(by_user.loc[2, "recall@3"], 0.0)
        self.assertAlmostEqual(by_user.loc[2, "hit_rate@3"], 0.0)
        self.assertAlmostEqual(by_user.loc[2, "map@3"], 0.0)
        self.assertAlmostEqual(by_user.loc[2, "ndcg@3"], 0.0)


if __name__ == "__main__":
    unittest.main()
