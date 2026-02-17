import json
import unittest

from config.paths import (
    DOCS_DATA_DIR,
    DOCS_DIR,
    DOCS_PROJECTION_PCA_JSON,
    RAW_EMBEDDINGS_NPZ,
    RAW_PROGRAM_CATEGORIZED_CSV,
    RAW_PROGRAM_CSV,
)
from server import app


class StructureSmokeTests(unittest.TestCase):
    def test_core_directories_exist(self):
        self.assertTrue(DOCS_DIR.exists())
        self.assertTrue(DOCS_DATA_DIR.exists())

    def test_core_data_files_exist(self):
        self.assertTrue(RAW_PROGRAM_CSV.exists())
        self.assertTrue(RAW_PROGRAM_CATEGORIZED_CSV.exists())
        self.assertTrue(RAW_EMBEDDINGS_NPZ.exists())

    def test_docs_entrypoints_exist(self):
        self.assertTrue((DOCS_DIR / "index.html").exists())
        self.assertTrue((DOCS_DIR / "viz" / "index.html").exists())
        self.assertTrue((DOCS_DIR / "assets" / "js" / "unified_app.js").exists())

    def test_projection_artifact_is_valid_json(self):
        self.assertTrue(DOCS_PROJECTION_PCA_JSON.exists())
        payload = json.loads(DOCS_PROJECTION_PCA_JSON.read_text(encoding="utf-8"))
        self.assertIn("points", payload)
        self.assertIsInstance(payload["points"], list)
        self.assertGreater(len(payload["points"]), 0)


class FlaskSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_root_route(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        res.close()

    def test_meta_route(self):
        res = self.client.get("/api/meta")
        self.assertEqual(res.status_code, 200)
        res.close()

    def test_docs_projection_data_route(self):
        res = self.client.get("/data/projection_pca.json")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn("points", payload)
        self.assertIsInstance(payload["points"], list)
        self.assertGreater(len(payload["points"]), 0)
        res.close()

    def test_docs_program_csv_data_route(self):
        res = self.client.get("/data/EACL_2026_program_categorized.csv")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/csv", res.content_type)
        body = res.get_data(as_text=True)
        self.assertIn("Paper", body)
        res.close()

    def test_viz_route(self):
        res = self.client.get("/viz")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.content_type)
        res.close()

    def test_topic_match_route(self):
        res = self.client.post("/api/topic_match", json={"ratings": {}, "slots": []})
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload.get("ok"))
        res.close()


if __name__ == "__main__":
    unittest.main()
