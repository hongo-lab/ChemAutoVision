import ast
import unittest
from pathlib import Path


class GraphTrainingChempropCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = (
            Path(__file__).resolve().parents[1] / "training" / "graph_training.py"
        )
        cls.tree = ast.parse(cls.source_path.read_text(encoding="utf-8"))

    def calls_named(self, function_name):
        return [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function_name
        ]

    def test_uses_chemprop_1_6_bond_descriptor_names(self):
        source = self.source_path.read_text(encoding="utf-8")

        incompatible_names = {
            "bond_features_path",
            "separate_test_bond_features_path",
            "separate_val_bond_features_path",
            "bond_feature_scaling",
            "bond_features_size",
            "scale_bond_features",
        }
        for name in incompatible_names:
            self.assertNotIn(name, source)

        self.assertIn("bond_descriptors_path", source)
        self.assertIn("separate_test_bond_descriptors_path", source)
        self.assertIn("separate_val_bond_descriptors_path", source)
        self.assertIn("scale_bond_descriptors", source)

    def test_checkpoint_calls_include_atom_bond_scaler(self):
        calls = self.calls_named("save_checkpoint")

        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            positional_names = [
                argument.id
                for argument in call.args
                if isinstance(argument, ast.Name)
            ]
            self.assertIn("atom_bond_scaler", positional_names)
            self.assertEqual(len(call.args), 8)

    def test_train_evaluate_and_predict_receive_atom_bond_scaler(self):
        for function_name in ("train", "evaluate", "predict"):
            calls = self.calls_named(function_name)
            self.assertGreaterEqual(len(calls), 1)
            for call in calls:
                keyword_names = {keyword.arg for keyword in call.keywords}
                self.assertIn("atom_bond_scaler", keyword_names)

    def test_prediction_evaluation_receives_atom_bond_target_flag(self):
        calls = self.calls_named("evaluate_predictions")

        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            keyword_names = {keyword.arg for keyword in call.keywords}
            self.assertIn("is_atom_bond_targets", keyword_names)


if __name__ == "__main__":
    unittest.main()
