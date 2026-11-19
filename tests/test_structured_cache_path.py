import json
import tempfile
import unittest
from pathlib import Path

from src.structured_features import cache_key, resolve_cache_directory


class TestStructuredCachePath(unittest.TestCase):
    def test_resolves_exact_keyed_cache_directory_without_nesting_key_twice(self):
        config = {"text_checkpoint": "text", "text_revision": "t1",
                  "image_checkpoint": "image", "image_revision": "i1",
                  "max_length": 256, "image_size": 224}
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "tmp") as temp:
            exact = Path(temp) / cache_key(config)
            exact.mkdir()
            (exact / "manifest.json").write_text(json.dumps({"cache_key": cache_key(config)}))
            self.assertEqual(resolve_cache_directory(exact, config), exact.resolve())

    def test_resolves_cache_root_to_keyed_directory(self):
        config = {"text_checkpoint": "text", "text_revision": "t1",
                  "image_checkpoint": "image", "image_revision": "i1",
                  "max_length": 256, "image_size": 224}
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "tmp") as temp:
            root = Path(temp)
            exact = root / cache_key(config)
            exact.mkdir()
            (exact / "manifest.json").write_text(json.dumps({"cache_key": cache_key(config)}))
            self.assertEqual(resolve_cache_directory(root, config), exact.resolve())


if __name__ == "__main__":
    unittest.main()
