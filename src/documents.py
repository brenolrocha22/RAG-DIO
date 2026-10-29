"""Monta os textos indexados pelo RAG a partir dos três CSVs.

Este módulo não depende do LlamaIndex nem do Chainlit, para que possa ser
testado isoladamente. Há dois tipos de registro:

- "venda": um texto por venda, com veículo, data, preço e região do cliente;
- "resumo": totais pré-calculados (por marca, modelo, país, região e ano, além
  de rankings), que permitem responder perguntas agregadas que a busca por
  similaridade sobre vendas individuais não consegue responder.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Hashable, Iterable, TypeVar


K = TypeVar("K", bound=Hashable)


SALES_COLUMNS = {"sale_id", "vehicle_id", "customer_id", "data_venda", "preco"}
VEHICLE_COLUMNS = {"vehicle_id", "marca", "modelo"}
CUSTOMER_COLUMNS = {"customer_id", "pais", "regiao"}


@dataclass(frozen=True)
class Sale:
    sale_id: str
    vehicle_id: str
    customer_id: str
    data_venda: str
    preco: Decimal
    marca: str
    modelo: str
    pais: str
    regiao: str

    @property
    def ano(self) -> str:
        return self.data_venda[:4]


@dataclass(frozen=True)
class Record:
    text: str
    metadata: dict[str, str]


def read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV obrigatório não encontrado: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"O arquivo {path.name} não contém cabeçalho.")
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Colunas ausentes em {path.name}: {', '.join(sorted(missing_columns))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"O arquivo {path.name} não contém registros.")
    return rows


def load_sales(data_dir: Path) -> list[Sale]:
    sales = read_csv(data_dir / "vendas.csv", SALES_COLUMNS)
    vehicles = read_csv(data_dir / "veiculos.csv", VEHICLE_COLUMNS)
    customers = read_csv(data_dir / "clientes.csv", CUSTOMER_COLUMNS)
    vehicle_by_id = {row["vehicle_id"]: row for row in vehicles}
    customer_by_id = {row["customer_id"]: row for row in customers}

    joined = []
    for sale in sales:
        vehicle = vehicle_by_id.get(sale["vehicle_id"])
        customer = customer_by_id.get(sale["customer_id"])
        if vehicle is None:
            raise ValueError(
                f"A venda {sale['sale_id']} referencia veículo inexistente "
                f"{sale['vehicle_id']}."
            )
        if customer is None:
            raise ValueError(
                f"A venda {sale['sale_id']} referencia cliente inexistente "
                f"{sale['customer_id']}."
            )
        joined.append(
            Sale(
                sale_id=sale["sale_id"],
                vehicle_id=sale["vehicle_id"],
                customer_id=sale["customer_id"],
                data_venda=sale["data_venda"],
                preco=Decimal(sale["preco"]),
                marca=vehicle["marca"],
                modelo=vehicle["modelo"],
                pais=customer["pais"],
                regiao=customer["regiao"],
            )
        )
    return joined


def money(value: Decimal) -> str:
    """Formata no padrão brasileiro: 1234567.8 -> 1.234.567,80."""
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def sale_records(sales: Iterable[Sale]) -> list[Record]:
    return [
        Record(
            text=(
                f"Venda: {sale.sale_id}\n"
                f"Veículo: {sale.marca} {sale.modelo}\n"
                f"Marca: {sale.marca}\n"
                f"Data da venda: {sale.data_venda}\n"
                f"Preço: {money(sale.preco)}\n"
                f"País do cliente: {sale.pais}\n"
                f"Região do cliente: {sale.regiao}"
            ),
            metadata={
                "tipo": "venda",
                "sale_id": sale.sale_id,
                "vehicle_id": sale.vehicle_id,
                "customer_id": sale.customer_id,
            },
        )
        for sale in sales
    ]


def _group(sales: Iterable[Sale], key: Callable[[Sale], K]) -> dict[K, list[Sale]]:
    groups: dict[K, list[Sale]] = defaultdict(list)
    for sale in sales:
        groups[key(sale)].append(sale)
    return groups


def _totals(sales: list[Sale]) -> str:
    revenue = sum((sale.preco for sale in sales), Decimal())
    average = revenue / len(sales)
    first = min(sale.data_venda for sale in sales)
    last = max(sale.data_venda for sale in sales)
    return (
        f"Quantidade de vendas: {len(sales)}\n"
        f"Faturamento total: {money(revenue)}\n"
        f"Preço médio: {money(average)}\n"
        f"Período: {first} a {last}"
    )


def _breakdown(sales: list[Sale], key: Callable[[Sale], str], label: str) -> str:
    groups = _group(sales, key)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    lines = [
        f"- {name}: {len(items)} vendas, faturamento {money(sum((s.preco for s in items), Decimal()))}"
        for name, items in ordered
    ]
    return f"{label}:\n" + "\n".join(lines)


def _ranking(sales: list[Sale], key: Callable[[Sale], str], subject: str, answers: str) -> Record:
    groups = _group(sales, key)
    by_count = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    by_revenue = sorted(
        groups.items(),
        key=lambda item: (-sum((s.preco for s in item[1]), Decimal()), item[0]),
    )
    count_lines = [
        f"{position}. {name}: {len(items)} vendas"
        for position, (name, items) in enumerate(by_count, start=1)
    ]
    revenue_lines = [
        f"{position}. {name}: {money(sum((s.preco for s in items), Decimal()))}"
        for position, (name, items) in enumerate(by_revenue, start=1)
    ]
    return Record(
        text=(
            f"Resumo: ranking de {subject} (base completa). Responde {answers}.\n"
            f"Ranking por quantidade de vendas (mais vendido primeiro):\n"
            + "\n".join(count_lines)
            + "\nRanking por faturamento:\n"
            + "\n".join(revenue_lines)
        ),
        metadata={"tipo": "resumo", "escopo": f"ranking_{subject}"},
    )


def _price_summary(sales: list[Sale], limit: int = 10) -> Record:
    by_price = sorted(sales, key=lambda s: (s.preco, s.sale_id))
    groups = _group(sales, lambda s: f"{s.marca} {s.modelo}")
    averages = sorted(
        ((name, sum((s.preco for s in items), Decimal()) / len(items)) for name, items in groups.items()),
        key=lambda item: (-item[1], item[0]),
    )

    def sale_line(sale: Sale) -> str:
        return f"- {sale.sale_id}: {sale.marca} {sale.modelo}, {money(sale.preco)}, {sale.data_venda}, {sale.pais}"

    return Record(
        text=(
            "Resumo: preços (base completa). Responde qual foi a venda mais cara, "
            "a mais barata e qual modelo tem o maior preço médio.\n"
            f"Vendas mais caras:\n" + "\n".join(sale_line(s) for s in reversed(by_price[-limit:]))
            + "\nVendas mais baratas:\n" + "\n".join(sale_line(s) for s in by_price[:limit])
            + "\nPreço médio por modelo (maior primeiro):\n"
            + "\n".join(f"{position}. {name}: {money(average)}"
                        for position, (name, average) in enumerate(averages, start=1))
        ),
        metadata={"tipo": "resumo", "escopo": "precos"},
    )


def summary_records(sales: list[Sale]) -> list[Record]:
    if not sales:
        return []

    records = [
        Record(
            text=(
                "Resumo geral da base de vendas de carros\n"
                + _totals(sales)
                + f"\nMarcas diferentes: {len({s.marca for s in sales})}\n"
                f"Modelos diferentes: {len({s.modelo for s in sales})}\n"
                f"Países diferentes: {len({s.pais for s in sales})}\n"
                f"Regiões diferentes: {len({s.regiao for s in sales})}"
            ),
            metadata={"tipo": "resumo", "escopo": "geral"},
        ),
        _ranking(sales, lambda s: s.marca, "marcas",
                 "qual marca mais vendeu e qual marca teve maior faturamento"),
        _ranking(sales, lambda s: f"{s.marca} {s.modelo}", "modelos",
                 "qual modelo de carro mais vendeu e qual modelo teve maior faturamento"),
        _ranking(sales, lambda s: s.pais, "países",
                 "qual país comprou mais carros e qual país teve maior faturamento"),
        _ranking(sales, lambda s: s.ano, "anos",
                 "qual ano teve mais vendas e como as vendas evoluíram ano a ano"),
        _price_summary(sales),
    ]

    for marca, items in sorted(_group(sales, lambda s: s.marca).items()):
        records.append(
            Record(
                text=(
                    f"Resumo da marca {marca}: total de vendas, faturamento, "
                    f"modelos vendidos, países e anos da {marca}\n"
                    + _totals(items)
                    + "\n"
                    + _breakdown(items, lambda s: s.modelo, "Modelos")
                    + "\n"
                    + _breakdown(items, lambda s: s.pais, "Países")
                    + "\n"
                    + _breakdown(items, lambda s: s.ano, "Anos")
                ),
                metadata={"tipo": "resumo", "escopo": "marca", "valor": marca},
            )
        )

    for (marca, modelo), items in sorted(_group(sales, lambda s: (s.marca, s.modelo)).items()):
        records.append(
            Record(
                text=(
                    f"Resumo do modelo {modelo} (marca {marca})\n"
                    + _totals(items)
                    + "\n"
                    + _breakdown(items, lambda s: s.pais, "Países")
                    + "\n"
                    + _breakdown(items, lambda s: s.ano, "Anos")
                ),
                metadata={"tipo": "resumo", "escopo": "modelo", "valor": modelo},
            )
        )

    for pais, items in sorted(_group(sales, lambda s: s.pais).items()):
        records.append(
            Record(
                text=(
                    f"Resumo do país {pais}\n"
                    + _totals(items)
                    + "\n"
                    + _breakdown(items, lambda s: s.marca, "Marcas")
                    + "\n"
                    + _breakdown(items, lambda s: s.regiao, "Regiões")
                    + "\n"
                    + _breakdown(items, lambda s: s.ano, "Anos")
                ),
                metadata={"tipo": "resumo", "escopo": "pais", "valor": pais},
            )
        )

    for (pais, regiao), items in sorted(_group(sales, lambda s: (s.pais, s.regiao)).items()):
        records.append(
            Record(
                text=(
                    f"Resumo da região {regiao} ({pais})\n"
                    + _totals(items)
                    + "\n"
                    + _breakdown(items, lambda s: s.marca, "Marcas")
                ),
                metadata={"tipo": "resumo", "escopo": "regiao", "valor": regiao},
            )
        )

    for ano, items in sorted(_group(sales, lambda s: s.ano).items()):
        records.append(
            Record(
                text=(
                    f"Resumo do ano {ano}\n"
                    + _totals(items)
                    + "\n"
                    + _breakdown(items, lambda s: s.marca, "Marcas")
                    + "\n"
                    + _breakdown(items, lambda s: s.pais, "Países")
                ),
                metadata={"tipo": "resumo", "escopo": "ano", "valor": ano},
            )
        )

    return records
