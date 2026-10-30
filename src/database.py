"""Carrega os três CSVs em um banco SQLite e executa consultas somente leitura.

O banco é a fonte para perguntas analíticas livres (filtros combinados,
agrupamentos e cálculos) que os resumos pré-calculados não cobrem. Os CSVs
continuam sendo a fonte versionada; o banco é recriado a partir deles.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from documents import CUSTOMER_COLUMNS, SALES_COLUMNS, VEHICLE_COLUMNS, read_csv


MAX_ROWS = 50
QUERY_TIMEOUT_SECONDS = 5

SCHEMA_SQL = """
CREATE TABLE veiculos (
    vehicle_id TEXT PRIMARY KEY,
    marca TEXT NOT NULL,
    modelo TEXT NOT NULL
);
CREATE TABLE clientes (
    customer_id TEXT PRIMARY KEY,
    pais TEXT NOT NULL,
    regiao TEXT NOT NULL
);
CREATE TABLE vendas (
    sale_id TEXT PRIMARY KEY,
    vehicle_id TEXT NOT NULL REFERENCES veiculos(vehicle_id),
    customer_id TEXT NOT NULL REFERENCES clientes(customer_id),
    data_venda TEXT NOT NULL,
    preco REAL NOT NULL
);
CREATE INDEX idx_vendas_vehicle ON vendas(vehicle_id);
CREATE INDEX idx_vendas_customer ON vendas(customer_id);
CREATE VIEW vendas_detalhadas AS
SELECT
    v.sale_id,
    v.data_venda,
    CAST(substr(v.data_venda, 1, 4) AS INTEGER) AS ano,
    CAST(substr(v.data_venda, 6, 2) AS INTEGER) AS mes,
    v.preco,
    ve.marca,
    ve.modelo,
    c.pais,
    c.regiao,
    v.customer_id,
    v.vehicle_id
FROM vendas v
JOIN veiculos ve ON ve.vehicle_id = v.vehicle_id
JOIN clientes c ON c.customer_id = v.customer_id;
"""

SCHEMA_DESCRIPTION = """Banco SQLite com as vendas de carros. Prefira a view vendas_detalhadas:
vendas_detalhadas(sale_id TEXT, data_venda TEXT 'AAAA-MM-DD', ano INTEGER, mes INTEGER,
                  preco REAL, marca TEXT, modelo TEXT, pais TEXT, regiao TEXT,
                  customer_id TEXT, vehicle_id TEXT)
Tabelas base: vendas(sale_id, vehicle_id, customer_id, data_venda, preco),
veiculos(vehicle_id, marca, modelo), clientes(customer_id, pais, regiao)."""


def build_database(data_dir: Path, db_path: Path) -> None:
    """Recria o banco a partir dos CSVs, de forma atômica."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = db_path.with_suffix(".tmp")
    temporary_path.unlink(missing_ok=True)

    vehicles = read_csv(data_dir / "veiculos.csv", VEHICLE_COLUMNS)
    customers = read_csv(data_dir / "clientes.csv", CUSTOMER_COLUMNS)
    sales = read_csv(data_dir / "vendas.csv", SALES_COLUMNS)

    connection = sqlite3.connect(temporary_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_SQL)
        connection.executemany(
            "INSERT INTO veiculos VALUES (?, ?, ?)",
            [(r["vehicle_id"], r["marca"], r["modelo"]) for r in vehicles],
        )
        connection.executemany(
            "INSERT INTO clientes VALUES (?, ?, ?)",
            [(r["customer_id"], r["pais"], r["regiao"]) for r in customers],
        )
        connection.executemany(
            "INSERT INTO vendas VALUES (?, ?, ?, ?, ?)",
            [
                (r["sale_id"], r["vehicle_id"], r["customer_id"], r["data_venda"], float(r["preco"]))
                for r in sales
            ],
        )
        connection.commit()
    finally:
        connection.close()
    temporary_path.replace(db_path)


def distinct_values(db_path: Path, column: str) -> list[str]:
    """Valores existentes de uma coluna da view, para orientar os filtros do LLM."""
    if column not in {"marca", "modelo", "pais", "regiao", "ano"}:
        raise ValueError(f"Coluna não permitida: {column}")
    with _read_only_connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT DISTINCT {column} FROM vendas_detalhadas ORDER BY {column}"
        ).fetchall()
    return [str(row[0]) for row in rows]


def run_select(db_path: Path, sql: str, max_rows: int = MAX_ROWS) -> str:
    """Executa uma única consulta SELECT em conexão somente leitura.

    Retorna as linhas em texto tabular (no máximo max_rows) ou uma mensagem de
    erro legível, para que o LLM possa corrigir a consulta e tentar de novo.
    """
    statement = sql.strip().rstrip(";").strip()
    if not statement.lower().startswith(("select", "with")):
        return "Erro: apenas consultas SELECT (ou WITH ... SELECT) são permitidas."

    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
    try:
        with _read_only_connection(db_path) as connection:
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            cursor = connection.execute(statement)
            columns = [description[0] for description in cursor.description or []]
            rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as error:
        return f"Erro ao executar a consulta: {error}"

    if not rows:
        return "A consulta não retornou linhas."
    truncated = len(rows) > max_rows
    lines = [" | ".join(columns)]
    lines += [" | ".join(_format_cell(value) for value in row) for row in rows[:max_rows]]
    if truncated:
        lines.append(f"(resultado truncado em {max_rows} linhas; agregue ou filtre mais)")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "NULL" if value is None else str(value)


class _read_only_connection:
    """Abre o banco em modo somente leitura: nenhuma consulta altera os dados."""

    def __init__(self, db_path: Path) -> None:
        if not db_path.is_file():
            raise FileNotFoundError(f"Banco não encontrado: {db_path}")
        self._uri = db_path.resolve().as_uri() + "?mode=ro"

    def __enter__(self) -> sqlite3.Connection:
        self._connection = sqlite3.connect(self._uri, uri=True)
        self._connection.execute("PRAGMA query_only = ON")
        return self._connection

    def __exit__(self, *exc_info: object) -> None:
        self._connection.close()
