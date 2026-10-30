import sys
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from database import build_database, distinct_values, run_select  # noqa: E402
from documents import load_sales  # noqa: E402


class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temporary_dir.name) / "vendas.db"
        build_database(PROJECT_DIR / "data", cls.db_path)
        cls.sales = load_sales(PROJECT_DIR / "data")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_dir.cleanup()

    def test_view_has_every_sale(self) -> None:
        result = run_select(self.db_path, "SELECT COUNT(*) FROM vendas_detalhadas")
        self.assertEqual(result.splitlines()[1], "985")

    def test_total_revenue_matches_csv(self) -> None:
        revenue = sum((s.preco for s in self.sales), Decimal())
        result = run_select(self.db_path, "SELECT ROUND(SUM(preco), 2) FROM vendas")
        self.assertEqual(result.splitlines()[1], f"{revenue:.2f}")

    def test_brand_ranking_matches_csv(self) -> None:
        best_brand, count = Counter(s.marca for s in self.sales).most_common(1)[0]
        result = run_select(
            self.db_path,
            "SELECT marca, COUNT(*) FROM vendas_detalhadas "
            "GROUP BY marca ORDER BY COUNT(*) DESC LIMIT 1",
        )
        self.assertEqual(result.splitlines()[1], f"{best_brand} | {count}")

    def test_combined_filters(self) -> None:
        expected = sum(1 for s in self.sales if s.marca == "Honda" and s.pais == "Brazil" and s.ano == "2018")
        result = run_select(
            self.db_path,
            "SELECT COUNT(*) FROM vendas_detalhadas "
            "WHERE marca = 'Honda' AND pais = 'Brazil' AND ano = 2018",
        )
        self.assertEqual(result.splitlines()[1], str(expected))

    def test_rejects_statements_other_than_select(self) -> None:
        result = run_select(self.db_path, "DELETE FROM vendas")
        self.assertTrue(result.startswith("Erro"))

    def test_connection_is_read_only(self) -> None:
        result = run_select(self.db_path, "WITH x AS (SELECT 1) DELETE FROM vendas")
        self.assertTrue(result.startswith("Erro"))
        count = run_select(self.db_path, "SELECT COUNT(*) FROM vendas")
        self.assertEqual(count.splitlines()[1], "985")

    def test_invalid_sql_returns_readable_error(self) -> None:
        result = run_select(self.db_path, "SELECT coluna_inexistente FROM vendas")
        self.assertIn("no such column", result)

    def test_results_are_truncated(self) -> None:
        result = run_select(self.db_path, "SELECT sale_id FROM vendas", max_rows=10)
        self.assertEqual(len(result.splitlines()), 12)
        self.assertIn("truncado", result.splitlines()[-1])

    def test_distinct_values_lists_brands(self) -> None:
        self.assertEqual(distinct_values(self.db_path, "marca"), sorted({s.marca for s in self.sales}))


if __name__ == "__main__":
    unittest.main()
