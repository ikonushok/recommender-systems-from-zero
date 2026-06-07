import unittest

import pandas as pd

from recsys_basics.basic.content_based import (
    ContentBasedRecommender,
    prepare_genre_text_features,
)


class ContentBasedRecommenderTest(unittest.TestCase):
    def setUp(self) -> None:
        movies = pd.DataFrame(
            {
                "item_id": [1, 2, 3, 4],
                "title": ["First", "Twin", "Romance", "Action"],
                "genres": [
                    "Action|Adventure",
                    "Action|Adventure",
                    "Romance|Drama",
                    "Action",
                ],
            }
        )
        self.items = prepare_genre_text_features(movies)
        self.model = ContentBasedRecommender().fit(self.items)

    def test_prepare_genres_builds_readable_text_features(self) -> None:
        self.assertEqual(
            self.items["genres_text"].tolist(),
            ["Action Adventure", "Action Adventure", "Romance Drama", "Action"],
        )

    def test_similar_items_exclude_query_item_and_rank_exact_match_first(self) -> None:
        similar = self.model.get_similar_items(item_id=1, k=2)

        self.assertEqual(similar.iloc[0]["item_id"], 2)
        self.assertNotIn(1, similar["item_id"].tolist())
        self.assertGreater(similar.iloc[0]["similarity"], similar.iloc[1]["similarity"])

    def test_recommend_filters_seen_items_and_has_no_duplicates(self) -> None:
        recommendations = self.model.recommend(
            history_item_ids=[1],
            seen_items={1, 2},
            k=3,
        )

        self.assertEqual(recommendations[0], 4)
        self.assertTrue({1, 2}.isdisjoint(recommendations))
        self.assertEqual(len(recommendations), len(set(recommendations)))

    def test_unknown_history_has_explicit_empty_fallback(self) -> None:
        self.assertEqual(
            self.model.recommend(history_item_ids=[999], seen_items=set(), k=3),
            [],
        )


if __name__ == "__main__":
    unittest.main()
