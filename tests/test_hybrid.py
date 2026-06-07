import unittest

import pandas as pd

from recsys_basics.basic.hybrid import (
    HybridInterleavingRecommender,
    HybridRankFusionRecommender,
    reciprocal_rank_fusion,
    weighted_interleave,
)


class HybridRecommenderTest(unittest.TestCase):
    def test_weighted_interleave_removes_duplicates_and_respects_top_k(self) -> None:
        ranked_lists = {
            "content": [10, 20, 30, 40],
            "collaborative": [20, 30, 50, 60],
        }

        recommendations = weighted_interleave(
            ranked_lists=ranked_lists,
            weights={"content": 1.0, "collaborative": 2.0},
            top_k=5,
        )

        self.assertEqual(len(recommendations), len(set(recommendations)))
        self.assertEqual(len(recommendations), 5)
        self.assertEqual(recommendations[0], 20)

    def test_rank_fusion_recommend_returns_unique_items(self) -> None:
        model = HybridRankFusionRecommender(weights={"content": 1.0, "collaborative": 1.0})

        recommendations = model.recommend(
            ranked_lists={
                "content": [10, 20, 30],
                "collaborative": [20, 40, 10],
            },
            top_k=4,
        )

        self.assertEqual(len(recommendations), len(set(recommendations)))
        self.assertEqual(recommendations, [20, 10, 40, 30])

    def test_recommend_many_outputs_ranked_rows_without_duplicate_items_per_user(self) -> None:
        model = HybridInterleavingRecommender(weights={"content": 1.0, "collaborative": 1.0})

        result = model.recommend_many(
            user_ranked_lists={
                1: {"content": [10, 20, 30], "collaborative": [20, 40, 10]},
                2: {"content": [50, 60], "collaborative": [60, 70]},
            },
            top_k=3,
        )

        self.assertEqual(sorted(result.columns.tolist()), ["item_id", "rank", "user_id"])
        for user_id, user_rows in result.groupby("user_id"):
            item_ids = user_rows["item_id"].tolist()
            self.assertEqual(len(item_ids), len(set(item_ids)), msg=f"duplicate item for user_id={user_id}")
            self.assertEqual(user_rows["rank"].tolist(), list(range(1, len(user_rows) + 1)))

    def test_reciprocal_rank_fusion_returns_stable_sorted_table(self) -> None:
        fused = reciprocal_rank_fusion(
            ranked_lists={
                "a": [10, 20],
                "b": [20, 30],
            }
        )

        self.assertIsInstance(fused, pd.DataFrame)
        self.assertEqual(fused.columns.tolist(), ["item_id", "hybrid_score"])
        self.assertEqual(fused["item_id"].tolist(), [20, 10, 30])


if __name__ == "__main__":
    unittest.main()
