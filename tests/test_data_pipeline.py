import csv
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


class CarSalesDataTests(unittest.TestCase):
    def test_three_csvs_have_expected_relationships(self) -> None:
        sales = read_csv("vendas.csv")
        vehicles = read_csv("veiculos.csv")
        customers = read_csv("clientes.csv")

        vehicle_ids = {row["vehicle_id"] for row in vehicles}
        customer_ids = {row["customer_id"] for row in customers}

        self.assertEqual(len(sales), 985)
        self.assertEqual(len(vehicles), 54)
        self.assertEqual(len(customers), 956)
        self.assertEqual(len({row["sale_id"] for row in sales}), len(sales))
        self.assertTrue(all(row["vehicle_id"] in vehicle_ids for row in sales))
        self.assertTrue(all(row["customer_id"] in customer_ids for row in sales))

    def test_customer_csv_does_not_contain_names(self) -> None:
        customers = read_csv("clientes.csv")
        self.assertEqual(set(customers[0]), {"customer_id", "pais", "regiao"})

    def test_sales_csv_has_iso_dates_and_decimal_prices(self) -> None:
        sales = read_csv("vendas.csv")
        self.assertRegex(sales[0]["data_venda"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(sales[0]["preco"], r"^\d+\.\d{2}$")


if __name__ == "__main__":
    unittest.main()

