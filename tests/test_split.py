import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from recsys_basics.split import (
    assert_no_user_time_leakage,
    leave_last_one_out_split,
    leave_last_two_out_split,
)


class TimeAwareSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.interactions = pd.DataFrame(
            {
                "user_id": [1, 2, 1, 3, 2, 1],
                "item_id": [12, 21, 10, 30, 20, 11],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-03",
                        "2024-01-02",
                        "2024-01-01",
                        "2024-01-01",
                        "2024-01-01",
                        "2024-01-02",
                    ]
                ),
            }
        )

    def test_leave_last_one_out_uses_future_interaction_for_test(self) -> None:
        split = leave_last_one_out_split(self.interactions)

        expected_train = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "item_id": [10, 11, 20],
                "timestamp": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-01"]
                ),
            }
        )
        expected_test = pd.DataFrame(
            {
                "user_id": [1, 2],
                "item_id": [12, 21],
                "timestamp": pd.to_datetime(["2024-01-03", "2024-01-02"]),
            }
        )

        assert_frame_equal(split.train, expected_train)
        assert_frame_equal(split.test, expected_test)
        assert_no_user_time_leakage(split.train, split.test)
        self.assertNotIn(3, split.train["user_id"].tolist())
        self.assertNotIn(3, split.test["user_id"].tolist())

    def test_leave_last_two_out_builds_train_validation_test(self) -> None:
        split = leave_last_two_out_split(self.interactions)

        self.assertEqual(split.train["item_id"].tolist(), [10])
        self.assertEqual(split.validation["item_id"].tolist(), [11])
        self.assertEqual(split.test["item_id"].tolist(), [12])
        assert_no_user_time_leakage(split.train, split.validation)
        assert_no_user_time_leakage(split.validation, split.test)

    def test_leakage_check_rejects_train_events_after_test(self) -> None:
        train = pd.DataFrame(
            {"user_id": [1], "timestamp": pd.to_datetime(["2024-01-03"])}
        )
        test = pd.DataFrame(
            {"user_id": [1], "timestamp": pd.to_datetime(["2024-01-02"])}
        )

        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_no_user_time_leakage(train, test)

    def test_split_validates_required_columns_and_minimum_activity(self) -> None:
        with self.assertRaisesRegex(ValueError, "timestamp"):
            leave_last_one_out_split(self.interactions.drop(columns="timestamp"))

        with self.assertRaisesRegex(ValueError, "не меньше 2"):
            leave_last_one_out_split(self.interactions, min_user_interactions=1)


if __name__ == "__main__":
    unittest.main()
