import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from recsys_basics.basic.popularity import (
    PopularityRecommender,
    build_seen_items_map,
    filter_positive_explicit_feedback,
)


class PopularityRecommenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.interactions = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 4],
                "item_id": [10, 20, 10, 20, 10, 30],
            }
        )

    def test_fit_ranks_items_by_count_with_stable_tie_break(self) -> None:
        model = PopularityRecommender().fit(self.interactions)

        expected = pd.DataFrame(
            {
                "item_id": [10, 20, 30],
                "popularity_score": [3, 2, 1],
            }
        )
        assert_frame_equal(model.get_popularity_table(), expected)

    def test_recommend_filters_seen_items_and_has_no_duplicates(self) -> None:
        model = PopularityRecommender().fit(self.interactions)

        recommendations = model.recommend(seen_items={10}, k=5)

        self.assertEqual(recommendations, [20, 30])
        self.assertEqual(len(recommendations), len(set(recommendations)))
        self.assertEqual(model.recommend(seen_items=set(), k=0), [])

    def test_recommend_many_supports_unknown_users_with_global_fallback(self) -> None:
        model = PopularityRecommender().fit(self.interactions)

        result = model.recommend_many(
            user_ids=[1, 999],
            seen_items_map={1: {10, 20}},
            k=2,
        )

        self.assertEqual(result[result["user_id"] == 1]["item_id"].tolist(), [30])
        self.assertEqual(
            result[result["user_id"] == 999]["item_id"].tolist(),
            [10, 20],
        )

    def test_feedback_filter_and_seen_items_map_preserve_meaning(self) -> None:
        explicit = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "item_id": [10, 20, 10],
                "rating": [5.0, 3.0, 4.0],
                "timestamp": pd.to_datetime(
                    ["2024-01-02", "2024-01-01", "2024-01-03"]
                ),
            }
        )

        positive = filter_positive_explicit_feedback(explicit, min_rating=4.0)

        self.assertEqual(positive["item_id"].tolist(), [10, 10])
        self.assertEqual(positive["event"].unique().tolist(), ["positive_rating"])
        self.assertEqual(build_seen_items_map(self.interactions)[1], {10, 20})


if __name__ == "__main__":
    unittest.main()
