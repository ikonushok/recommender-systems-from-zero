import unittest

import pandas as pd

from recsys_basics.basic.item_item import ItemItemRecommender


class ItemItemRecommenderTest(unittest.TestCase):
    def setUp(self) -> None:
        interactions = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "item_id": [1, 2, 1, 2, 1, 3, 3, 4],
            }
        )
        self.model = ItemItemRecommender().fit(interactions)

    def test_fit_deduplicates_interactions_and_preserves_id_mapping(self) -> None:
        duplicated = pd.concat(
            [
                self.model.interactions_,
                pd.DataFrame({"user_id": [1], "item_id": [2]}),
            ],
            ignore_index=True,
        )
        model = ItemItemRecommender().fit(duplicated)

        self.assertEqual(len(model.interactions_), 8)
        self.assertEqual(set(model.item_to_index_), {1, 2, 3, 4})
        self.assertEqual(set(model.index_to_item_.values()), {1, 2, 3, 4})

    def test_similar_items_rank_stronger_cooccurrence_first(self) -> None:
        similar = self.model.get_similar_items(item_id=1, k=2)

        self.assertEqual(similar["item_id"].tolist(), [2, 3])
        self.assertGreater(similar.iloc[0]["similarity"], similar.iloc[1]["similarity"])

    def test_recommend_filters_seen_items_and_has_no_duplicates(self) -> None:
        recommendations = self.model.recommend(
            history_item_ids=[1],
            seen_items={1, 2},
            k=3,
        )

        self.assertEqual(recommendations, [3])
        self.assertTrue({1, 2}.isdisjoint(recommendations))
        self.assertEqual(len(recommendations), len(set(recommendations)))

    def test_unknown_history_has_explicit_empty_fallback(self) -> None:
        self.assertEqual(
            self.model.recommend(history_item_ids=[999], seen_items=set(), k=3),
            [],
        )


if __name__ == "__main__":
    unittest.main()
