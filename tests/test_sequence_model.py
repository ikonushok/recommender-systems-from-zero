import unittest

import pandas as pd

from recsys_basics.advanced.sequence_model import (
    LastItemTransitionRecommender,
    build_user_last_items,
)


class LastItemTransitionRecommenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.interactions = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2, 3, 3],
                "item_id": [10, 20, 30, 10, 20, 40, 20, 50],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-01",
                        "2024-01-02",
                    ]
                ),
            }
        )
        self.model = LastItemTransitionRecommender().fit(self.interactions)

    def test_build_user_last_items_returns_latest_item_per_user(self) -> None:
        last_items = build_user_last_items(self.interactions)

        self.assertEqual(last_items, {1: 30, 2: 40, 3: 50})

    def test_fit_recommend_filters_seen_items_and_has_no_duplicates(self) -> None:
        recommendations = self.model.recommend(
            last_item_id=10,
            seen_items={10, 20},
            k=5,
        )

        self.assertEqual(recommendations, [30, 40, 50])
        self.assertEqual(len(recommendations), len(set(recommendations)))
        self.assertTrue({10, 20}.isdisjoint(recommendations))

    def test_recommend_many_returns_ranked_rows_without_duplicate_items_per_user(self) -> None:
        result = self.model.recommend_many(
            user_last_items={1: 10, 2: 20},
            seen_items_map={1: {10}, 2: {20}},
            k=3,
        )

        for user_id, user_rows in result.groupby("user_id"):
            item_ids = user_rows["item_id"].tolist()
            self.assertEqual(len(item_ids), len(set(item_ids)), msg=f"duplicate item for user_id={user_id}")
            self.assertEqual(user_rows["rank"].tolist(), list(range(1, len(user_rows) + 1)))

    def test_unknown_last_item_falls_back_to_global_counts(self) -> None:
        recommendations = self.model.recommend(last_item_id=999, seen_items={20}, k=3)

        self.assertEqual(recommendations, [10, 30, 40])


if __name__ == "__main__":
    unittest.main()
