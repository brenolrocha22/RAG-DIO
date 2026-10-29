"""Gera a imagem de visão geral das vendas usada no README.

Uso (requer matplotlib, que não faz parte das dependências do app):

    python -m pip install matplotlib
    python scripts/gerar_graficos.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from documents import load_sales, money  # noqa: E402

OUTPUT = PROJECT_DIR / "docs" / "img" / "visao-geral-vendas.png"
ACCENT = "#c8102e"
NEUTRAL = "#9aa0a6"
TEXT = "#202124"


def totals(sales, key):
    revenue: dict[str, Decimal] = defaultdict(Decimal)
    count: dict[str, int] = defaultdict(int)
    for sale in sales:
        revenue[key(sale)] += sale.preco
        count[key(sale)] += 1
    return revenue, count


def millions(value, _position):
    return f"{value / 1_000_000:.0f} mi"


def main() -> None:
    sales = load_sales(PROJECT_DIR / "data")
    total = sum((s.preco for s in sales), Decimal())

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT})
    fig = plt.figure(figsize=(13, 7.5), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 3], hspace=0.45, wspace=0.35)

    kpis = [
        ("Faturamento total", money(total)),
        ("Vendas", f"{len(sales)}"),
        ("Ticket médio", money(total / len(sales))),
    ]
    for position, (label, value) in enumerate(kpis):
        axis = fig.add_subplot(grid[0, position])
        axis.axis("off")
        axis.text(0, 0.65, label, fontsize=11, color="#5f6368")
        axis.text(0, 0.1, value, fontsize=20, fontweight="bold")

    revenue, count = totals(sales, lambda s: s.marca)
    brands = sorted(count, key=count.get)
    axis = fig.add_subplot(grid[1, 0])
    axis.barh(brands, [count[b] for b in brands],
              color=[ACCENT if b == brands[-1] else NEUTRAL for b in brands])
    axis.set_title("Vendas por marca", loc="left", fontweight="bold")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    revenue, _ = totals(sales, lambda s: s.pais)
    countries = sorted(revenue, key=revenue.get)
    axis = fig.add_subplot(grid[1, 1])
    axis.barh(countries, [float(revenue[c]) for c in countries],
              color=[ACCENT if c == countries[-1] else NEUTRAL for c in countries])
    axis.xaxis.set_major_formatter(FuncFormatter(millions))
    axis.set_title("Faturamento por país", loc="left", fontweight="bold")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    revenue, _ = totals(sales, lambda s: s.ano)
    years = sorted(revenue)
    axis = fig.add_subplot(grid[1, 2])
    axis.plot(years, [float(revenue[y]) for y in years], color=ACCENT, marker="o", linewidth=2)
    axis.yaxis.set_major_formatter(FuncFormatter(millions))
    axis.set_ylim(bottom=0)
    axis.tick_params(axis="x", labelrotation=45)
    axis.set_title("Faturamento por ano", loc="left", fontweight="bold")
    axis.annotate("2020: dados até junho", xy=(years[-1], float(revenue[years[-1]])),
                  xytext=(-110, -10), textcoords="offset points", fontsize=8, color="#5f6368")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=110, bbox_inches="tight")
    print(f"Imagem gerada em {OUTPUT}")


if __name__ == "__main__":
    main()
