import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from callbacks import GraphEarlyStopping
from recording.record_mlflow import record_exp_result
from training.early_stopping import (
    get_best_checkpoint_path,
    restore_best_checkpoint,
    update_early_stopping_and_save,
)


class GraphBestCheckpointTest(unittest.TestCase):
    def test_stopping_restores_checkpoint_from_best_mse_epoch(self):
        scores = [0.50, 0.40, 0.45, 0.48]
        early_stopping = GraphEarlyStopping(patience=2, mode="min")

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "trained" / "model_0" / "model.pt"
            )
            checkpoint_path.parent.mkdir(parents=True)

            for epoch, score in enumerate(scores):
                def save_best(current_epoch=epoch, current_score=score):
                    checkpoint_path.write_text(
                        f"epoch={current_epoch},mse={current_score}", encoding="utf-8"
                    )

                should_stop = update_early_stopping_and_save(
                    early_stopping=early_stopping,
                    current_score=score,
                    epoch=epoch,
                    save_best=save_best,
                )
                if should_stop:
                    break

            restored_checkpoint = restore_best_checkpoint(
                str(checkpoint_path),
                lambda path: Path(path).read_text(encoding="utf-8"),
            )

        self.assertTrue(early_stopping.should_stop)
        self.assertEqual(early_stopping.best_epoch, 1)
        self.assertEqual(early_stopping.stopped_epoch, 3)
        self.assertEqual(restored_checkpoint, "epoch=1,mse=0.4")

    def test_mlflow_model_path_is_the_restored_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "trained" / "model_0" / "model.pt"
            )
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_text("best model", encoding="utf-8")

            early_stopping_results = [
                {
                    "model_idx": 0,
                    "best_epoch": 1,
                    "best_score": 0.40,
                    "checkpoint_path": str(checkpoint_path),
                }
            ]
            model_path = get_best_checkpoint_path(early_stopping_results)
            tags = {
                "target": "test_task",
                "model_name": "chemprops",
                "explanatory_val": "graph",
                "model_path": model_path,
            }

            with (
                patch("recording.record_mlflow.mlflow.set_tracking_uri"),
                patch(
                    "recording.record_mlflow.mlflow.start_run",
                    return_value=nullcontext(),
                ),
                patch("recording.record_mlflow.mlflow.set_tag") as set_tag,
                patch("recording.record_mlflow.mlflow.log_param"),
                patch("recording.record_mlflow.mlflow.log_metric"),
            ):
                record_exp_result(1, metrics={}, params={}, tags=tags)

            self.assertEqual(Path(model_path), checkpoint_path)
            self.assertTrue(checkpoint_path.is_file())
            set_tag.assert_any_call(key="model_path", value=str(checkpoint_path))


if __name__ == "__main__":
    unittest.main()
