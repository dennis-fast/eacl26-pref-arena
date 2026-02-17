import json
import unittest

from config.paths import DOCS_DIR, DOCS_PROJECTION_PCA_JSON, LEGACY_WEB_DIR


class StaticVizContractTests(unittest.TestCase):
    @staticmethod
    def _load_projection_payload() -> dict:
        return json.loads(DOCS_PROJECTION_PCA_JSON.read_text(encoding="utf-8"))

    @staticmethod
    def _apply_static_projection_like_filters(
        records: list[dict],
        mode: str = "all",
        session_value: str | None = None,
        oral_only: bool = False,
        search_term: str | None = None,
    ) -> list[dict]:
        rows = list(records)
        if mode == "session" and session_value is not None:
            rows = [row for row in rows if str(row.get("session", "")) == session_value]

        if oral_only:
            rows = [
                row
                for row in rows
                if "oral" in str(row.get("type_presentation", "")).lower()
                and str(row.get("attendance_type", "")).strip().lower() == "in-person"
            ]

        if search_term:
            needle = search_term.lower()
            rows = [row for row in rows if needle in str(row.get("title", "")).lower()]

        return rows

    def test_projection_artifact_backend_free_contract(self):
        payload = self._load_projection_payload()

        self.assertIn("points_meta", payload)
        self.assertIn("methods", payload)
        self.assertIn("neighbors", payload)
        self.assertIn("filters", payload)
        self.assertIn("available_sessions", payload)

        points_meta = payload["points_meta"]
        methods = payload["methods"]
        neighbors = payload["neighbors"]
        self.assertIsInstance(points_meta, list)
        self.assertIsInstance(methods, dict)
        self.assertIsInstance(neighbors, dict)
        self.assertGreater(len(points_meta), 0)

        for method_name in ["pca", "tsne", "umap"]:
            self.assertIn(method_name, methods)
            self.assertIsInstance(methods[method_name], list)
            self.assertGreater(len(methods[method_name]), 0)

        required_meta_fields = {
            "paper_id",
            "title",
            "session",
            "room_location",
            "type_presentation",
            "attendance_type",
        }
        for row in points_meta[:5]:
            self.assertTrue(required_meta_fields.issubset(set(row.keys())))

        required_coord_fields = {"paper_id", "x", "y"}
        for method_name in ["pca", "tsne", "umap"]:
            for row in methods[method_name][:5]:
                self.assertTrue(required_coord_fields.issubset(set(row.keys())))
                self.assertIsInstance(row["x"], (int, float))
                self.assertIsInstance(row["y"], (int, float))

        ids_meta = {str(row["paper_id"]) for row in points_meta}
        ids_pca = {str(row["paper_id"]) for row in methods["pca"]}
        ids_tsne = {str(row["paper_id"]) for row in methods["tsne"]}
        ids_umap = {str(row["paper_id"]) for row in methods["umap"]}

        self.assertEqual(ids_meta, ids_pca)
        self.assertEqual(ids_meta, ids_tsne)
        self.assertEqual(ids_meta, ids_umap)

        any_neighbor_key = next(iter(neighbors.keys()))
        self.assertIn(any_neighbor_key, ids_meta)
        self.assertIsInstance(neighbors[any_neighbor_key], list)
        if neighbors[any_neighbor_key]:
            required_neighbor_fields = {
                "paper_id",
                "title",
                "cosine",
                "session",
                "room_location",
                "type_presentation",
                "attendance_type",
            }
            self.assertTrue(required_neighbor_fields.issubset(set(neighbors[any_neighbor_key][0].keys())))

    def test_viz_apps_do_not_require_projection_apis(self):
        docs_viz_app = DOCS_DIR / "viz" / "app.js"
        legacy_viz_app = LEGACY_WEB_DIR / "static" / "app.js"
        self.assertTrue(docs_viz_app.exists())
        self.assertTrue(legacy_viz_app.exists())

        blocked_api_calls = ["/api/meta", "/api/project", "/api/nn"]
        required_static_marker = "projection_pca.json"

        docs_src = docs_viz_app.read_text(encoding="utf-8")
        legacy_src = legacy_viz_app.read_text(encoding="utf-8")

        self.assertIn(required_static_marker, docs_src)
        self.assertIn(required_static_marker, legacy_src)

        for api_path in blocked_api_calls:
            self.assertNotIn(api_path, docs_src)
            self.assertNotIn(api_path, legacy_src)

    def test_static_projection_session_oral_search_pipeline(self):
        payload = self._load_projection_payload()
        records = payload["points_meta"]
        pca_ids = {str(row["paper_id"]) for row in payload["methods"]["pca"]}

        first = records[0]
        session_value = str(first["session"])
        first_title = str(first["title"])
        search_term = next((tok for tok in first_title.split() if len(tok) >= 4), first_title[:4]).lower()

        session_rows = self._apply_static_projection_like_filters(
            records,
            mode="session",
            session_value=session_value,
        )
        expected_session_rows = [row for row in records if str(row["session"]) == session_value]
        self.assertEqual(len(session_rows), len(expected_session_rows))
        self.assertGreater(len(session_rows), 0)

        oral_rows = self._apply_static_projection_like_filters(
            session_rows,
            oral_only=True,
        )
        expected_oral_rows = [
            row
            for row in expected_session_rows
            if "oral" in str(row["type_presentation"]).lower()
            and str(row["attendance_type"]).strip().lower() == "in-person"
        ]
        self.assertEqual(len(oral_rows), len(expected_oral_rows))

        search_rows = self._apply_static_projection_like_filters(
            oral_rows,
            search_term=search_term,
        )
        expected_search_rows = [
            row for row in expected_oral_rows if search_term in str(row["title"]).lower()
        ]
        self.assertEqual(len(search_rows), len(expected_search_rows))

        for row in search_rows:
            self.assertIn(str(row["paper_id"]), pca_ids)

    def test_static_neighbors_subset_and_k_behavior(self):
        payload = self._load_projection_payload()
        neighbors = payload["neighbors"]

        paper_id = next((pid for pid, rows in neighbors.items() if len(rows) >= 5), None)
        if paper_id is None:
            self.skipTest("No paper has enough neighbors for subset/k behavior test.")

        rows = neighbors[paper_id]
        current_ids = {str(rows[1]["paper_id"]), str(rows[3]["paper_id"]), str(rows[4]["paper_id"])}
        k = 2
        filtered = [row for row in rows if str(row["paper_id"]) in current_ids][:k]

        self.assertLessEqual(len(filtered), k)
        self.assertGreater(len(filtered), 0)
        for row in filtered:
            self.assertIn(str(row["paper_id"]), current_ids)
            self.assertIsInstance(row["cosine"], (int, float))


if __name__ == "__main__":
    unittest.main()
