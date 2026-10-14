import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.embeddings import _cached_model_path


class EmbeddingCacheTests(unittest.TestCase):
    def test_uses_cached_hugging_face_snapshot_without_network_lookup(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = Path(temporary_directory) / "snapshot"
            snapshot.mkdir()
            modules_file = snapshot / "modules.json"
            modules_file.write_text("[]", encoding="utf-8")

            with patch(
                "huggingface_hub.try_to_load_from_cache",
                return_value=str(modules_file),
            ) as lookup:
                model_path, local_files_only = _cached_model_path(
                    "sentence-transformers/example"
                )

        lookup.assert_called_once_with(
            "sentence-transformers/example", "modules.json"
        )
        self.assertEqual(model_path, str(snapshot))
        self.assertTrue(local_files_only)

    def test_keeps_remote_download_available_when_model_is_not_cached(self):
        with patch(
            "huggingface_hub.try_to_load_from_cache", return_value=None
        ):
            model_path, local_files_only = _cached_model_path(
                "sentence-transformers/not-cached"
            )

        self.assertEqual(model_path, "sentence-transformers/not-cached")
        self.assertFalse(local_files_only)


if __name__ == "__main__":
    unittest.main()
