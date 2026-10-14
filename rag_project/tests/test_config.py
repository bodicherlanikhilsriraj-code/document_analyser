import os
import unittest
from unittest.mock import patch

from core.config import Settings
from core.errors import ConfigurationError


class ConfigurationTests(unittest.TestCase):
    def test_reads_valid_overrides(self):
        with patch.dict(os.environ, {"CHUNK_SIZE": "800", "CHUNK_OVERLAP": "100"}):
            settings = Settings.from_env()
        self.assertEqual(settings.chunk_size, 800)
        self.assertEqual(settings.chunk_overlap, 100)

    def test_rejects_invalid_overlap(self):
        with patch.dict(os.environ, {"CHUNK_SIZE": "200", "CHUNK_OVERLAP": "200"}):
            with self.assertRaises(ConfigurationError):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
