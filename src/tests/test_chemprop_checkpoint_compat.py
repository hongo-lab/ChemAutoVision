import os
import unittest
from unittest.mock import patch

from training.chemprop_checkpoint_compat import trusted_chemprop_checkpoint_loading


class TrustedChempropCheckpointLoadingTest(unittest.TestCase):
    def test_enables_full_checkpoint_loading_only_inside_context(self):
        variable = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"

        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn(variable, os.environ)
            with trusted_chemprop_checkpoint_loading():
                self.assertEqual(os.environ[variable], "1")
            self.assertNotIn(variable, os.environ)

    def test_restores_existing_environment_value(self):
        variable = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"

        with patch.dict(os.environ, {variable: "0"}, clear=True):
            with trusted_chemprop_checkpoint_loading():
                self.assertEqual(os.environ[variable], "1")
            self.assertEqual(os.environ[variable], "0")

    def test_restores_environment_after_exception(self):
        variable = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "checkpoint failure"):
                with trusted_chemprop_checkpoint_loading():
                    raise RuntimeError("checkpoint failure")
            self.assertNotIn(variable, os.environ)


if __name__ == "__main__":
    unittest.main()
