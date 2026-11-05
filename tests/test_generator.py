import unittest

from hmda_synth.generator import generate_synthetic_data


class TestGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_rows = [
            {"loan_amount": 100000, "income": 55000, "action_taken": "approved"},
            {"loan_amount": 120000, "income": 62000, "action_taken": "approved"},
            {"loan_amount": 90000, "income": 50000, "action_taken": "denied"},
            {"loan_amount": 150000, "income": 70000, "action_taken": "approved"},
        ]

    def test_invalid_model_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_synthetic_data(self.sample_rows, model="invalid")

    def test_ctgan_path_returns_expected_shape_and_columns(self) -> None:
        result = generate_synthetic_data(self.sample_rows, model="ctgan", num_samples=6, random_state=1)
        self.assertEqual(list(result[0].keys()), list(self.sample_rows[0].keys()))
        self.assertEqual(len(result), 6)

    def test_tabddpm_path_returns_expected_shape_and_columns(self) -> None:
        result = generate_synthetic_data(self.sample_rows, model="tabddpm", num_samples=5, random_state=1)
        self.assertEqual(list(result[0].keys()), list(self.sample_rows[0].keys()))
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
