import ast
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]


def _source(name: str) -> str:
    return (SRC_DIR / name).read_text(encoding="utf-8")


def _recorded_metric_keys(source: str) -> set[str]:
    tree = ast.parse(source)
    record_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "record_exp_result"
    )
    metric_expression = record_call.args[1]
    dictionaries = (
        [metric_expression.body, metric_expression.orelse]
        if isinstance(metric_expression, ast.IfExp)
        else [metric_expression]
    )
    return {
        key.value
        for dictionary in dictionaries
        if isinstance(dictionary, ast.Dict)
        for key in dictionary.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


class ImageExperimentValidationOnlyTest(unittest.TestCase):
    def test_classifier_predicts_validation_and_logs_validation_metrics(self):
        source = _source("experiment_image_classifier.py")
        metric_keys = _recorded_metric_keys(source)

        self.assertIn("best_model.predict(val_ds)", source)
        self.assertNotIn("best_model.predict(test_ds)", source)
        self.assertNotIn("test_ds", source)
        self.assertTrue(
            {
                "val_accuracy",
                "val_recall",
                "val_precision",
                "val_roc_auc",
                "val_mcc",
                "val_f1",
            }.issubset(metric_keys)
        )
        self.assertNotIn("roc_auc", metric_keys)
        self.assertNotIn("acc", metric_keys)

    def test_regressor_predicts_validation_and_logs_validation_metrics(self):
        source = _source("experiment_image_regressor.py")
        metric_keys = _recorded_metric_keys(source)

        self.assertIn("best_model.predict(val_ds)", source)
        self.assertNotIn("best_model.predict(test_ds)", source)
        self.assertNotIn("test_ds", source)
        self.assertTrue(
            {"val_rmse", "val_mse", "val_mae", "val_r2"}.issubset(metric_keys)
        )
        self.assertNotIn("rmse", metric_keys)
        self.assertNotIn("mse", metric_keys)


class GraphExperimentValidationOnlyTest(unittest.TestCase):
    def test_graph_experiment_does_not_reference_external_test_data(self):
        source = _source("experiment_graph.py")
        metric_keys = _recorded_metric_keys(source)

        self.assertNotIn("test_data_path", source)
        self.assertNotIn("atom_desc_test_path", source)
        self.assertIn("'--test_path', val_data_path", source)
        self.assertIn("evaluate_test=False", source)
        self.assertTrue({"val_roc_auc", "val_rmse"}.issubset(metric_keys))
        self.assertNotIn("roc_auc", metric_keys)
        self.assertNotIn("rmse", metric_keys)

    def test_graph_training_keeps_historical_test_evaluation_as_default(self):
        tree = ast.parse(_source("training/graph_training.py"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_training_with_early_stopping"
        )

        keyword_defaults = dict(
            zip(
                [argument.arg for argument in function.args.kwonlyargs],
                function.args.kw_defaults,
            )
        )
        evaluate_test_default = keyword_defaults["evaluate_test"]
        self.assertIsInstance(evaluate_test_default, ast.Constant)
        self.assertIs(evaluate_test_default.value, True)


if __name__ == "__main__":
    unittest.main()
