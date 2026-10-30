import sys
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from documents import load_sales, money, sale_records, summary_records  # noqa: E402


class DocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sales = load_sales(PROJECT_DIR / "data")
        cls.summaries = summary_records(cls.sales)

    def summary(self, escopo: str, valor: str | None = None) -> str:
        for record in self.summaries:
            if record.metadata.get("escopo") == escopo and record.metadata.get("valor") == valor:
                return record.text
        self.fail(f"Resumo não encontrado: {escopo} {valor}")

    def test_money_uses_brazilian_format(self) -> None:
        self.assertEqual(money(Decimal("1234567.8")), "1.234.567,80")

    def test_one_sale_record_per_sale(self) -> None:
        records = sale_records(self.sales)
        self.assertEqual(len(records), 985)
        self.assertTrue(all(r.metadata["tipo"] == "venda" for r in records))
        self.assertIn("Venda: VEN-000001", records[0].text)

    def test_summaries_cover_every_dimension(self) -> None:
        scopes = Counter(r.metadata["escopo"] for r in self.summaries)
        self.assertEqual(scopes["marca"], len({s.marca for s in self.sales}))
        self.assertEqual(scopes["modelo"], len({(s.marca, s.modelo) for s in self.sales}))
        self.assertEqual(scopes["pais"], len({s.pais for s in self.sales}))
        self.assertEqual(scopes["regiao"], len({(s.pais, s.regiao) for s in self.sales}))
        self.assertEqual(scopes["ano"], len({s.ano for s in self.sales}))

    def test_general_summary_matches_dataset_totals(self) -> None:
        revenue = sum((s.preco for s in self.sales), Decimal())
        text = self.summary("geral")
        self.assertIn("Quantidade de vendas: 985", text)
        self.assertIn(f"Faturamento total: {money(revenue)}", text)

    def test_brand_ranking_lists_best_seller_first(self) -> None:
        best_brand, count = Counter(s.marca for s in self.sales).most_common(1)[0]
        text = self.summary("ranking_marcas")
        self.assertIn(f"1. {best_brand}: {count} vendas", text)

    def test_price_summary_starts_with_most_expensive_sale(self) -> None:
        most_expensive = max(self.sales, key=lambda s: s.preco)
        text = self.summary("precos")
        first_sale = text.split("Vendas mais caras:\n", 1)[1].splitlines()[0]
        self.assertIn(money(most_expensive.preco), first_sale)

    def test_brand_summary_counts_match(self) -> None:
        honda = [s for s in self.sales if s.marca == "Honda"]
        self.assertIn(f"Quantidade de vendas: {len(honda)}", self.summary("marca", "Honda"))


if __name__ == "__main__":
    unittest.main()
