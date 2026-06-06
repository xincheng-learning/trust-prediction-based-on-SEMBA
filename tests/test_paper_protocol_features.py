import unittest

from experiments.paper_protocol_features import build_rows_for_split_paper_batch


class PaperProtocolFeaturesTest(unittest.TestCase):
    def test_paper_batch_features_include_current_batch_before_prediction(self):
        events = [
            {"src": 1, "dst": 10, "t": 1, "y": 1},
            {"src": 2, "dst": 10, "t": 2, "y": 0},
            {"src": 3, "dst": 10, "t": 3, "y": 1},
        ]

        state = None
        rows, state = build_rows_for_split_paper_batch(events, state=state, batch_size=2)

        self.assertEqual(len(rows), 3)

        # The first two events are in the same batch. Paper protocol writes that
        # batch into history before making features for the batch.
        self.assertEqual(rows[0]["dst_in_pos"], 1.0)
        self.assertEqual(rows[0]["dst_in_neg"], 1.0)
        self.assertEqual(rows[1]["dst_in_pos"], 1.0)
        self.assertEqual(rows[1]["dst_in_neg"], 1.0)

        # The third event is alone in the second batch, so its own positive edge
        # is already included when its row is generated.
        self.assertEqual(rows[2]["dst_in_pos"], 2.0)
        self.assertEqual(rows[2]["dst_in_neg"], 1.0)

    def test_paper_batch_state_continues_across_splits_but_batches_reset(self):
        train_events = [
            {"src": 1, "dst": 10, "t": 1, "y": 1},
            {"src": 2, "dst": 10, "t": 2, "y": 0},
        ]
        val_events = [
            {"src": 3, "dst": 10, "t": 3, "y": 1},
        ]

        train_rows, state = build_rows_for_split_paper_batch(
            train_events,
            state=None,
            batch_size=1000,
        )
        val_rows, state = build_rows_for_split_paper_batch(
            val_events,
            state=state,
            batch_size=1000,
        )

        self.assertEqual(train_rows[0]["dst_in_pos"], 1.0)
        self.assertEqual(train_rows[0]["dst_in_neg"], 1.0)

        # Validation inherits train history, and its current batch is also
        # written before features are made.
        self.assertEqual(val_rows[0]["dst_in_pos"], 2.0)
        self.assertEqual(val_rows[0]["dst_in_neg"], 1.0)


if __name__ == "__main__":
    unittest.main()
