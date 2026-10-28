from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import chainlit as cl
import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.chroma import ChromaVectorStore


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
COLLECTION_NAME = "vendas_de_carros"
METADATA_PATH = STORAGE_DIR / "index_metadata.json"
CSV_FILES = {
    "sales": DATA_DIR / "vendas.csv",
    "vehicles": DATA_DIR / "veiculos.csv",
    "customers": DATA_DIR / "clientes.csv",
}

load_dotenv(BASE_DIR / ".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

QA_TEMPLATE = PromptTemplate(
    """Você é um assistente que consulta registros de vendas de carros.

Regras:
- Responda em português do Brasil.
- Use somente os registros fornecidos no contexto; não invente informações.
- Diferencie claramente os registros quando houver mais de uma venda.
- Não apresente contagens ou totais da base inteira com base apenas nos exemplos recuperados.
- Se a informação não estiver no contexto, diga que não a encontrou nos registros recuperados.

Contexto:
{context_str}

Pergunta:
{query_str}

Resposta:
"""
)


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"A variável {name} não está configurada. Copie .env.example para .env "
            "e informe sua chave do OpenRouter."
        )
    return value


def configure_models() -> None:
    Settings.llm = OpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        api_key=required_env("OPENROUTER_API_KEY"),
        context_window=8192,
        max_tokens=512,
        temperature=0.1,
    )
    Settings.embed_model = FastEmbedEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )


def dataset_hash() -> str:
    digest = hashlib.sha256()
    for path in CSV_FILES.values():
        if not path.is_file():
            raise FileNotFoundError(f"CSV obrigatório não encontrado: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
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


def load_sales_documents() -> list[Document]:
    sales = read_csv(
        CSV_FILES["sales"],
        {"sale_id", "vehicle_id", "customer_id", "data_venda", "preco"},
    )
    vehicles = read_csv(CSV_FILES["vehicles"], {"vehicle_id", "marca", "modelo"})
    customers = read_csv(
        CSV_FILES["customers"], {"customer_id", "pais", "regiao"}
    )
    vehicle_by_id = {row["vehicle_id"]: row for row in vehicles}
    customer_by_id = {row["customer_id"]: row for row in customers}

    documents = []
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
        text = (
            f"Venda: {sale['sale_id']}\n"
            f"Veículo: {vehicle['marca']} {vehicle['modelo']}\n"
            f"Data da venda: {sale['data_venda']}\n"
            f"Preço: {sale['preco']}\n"
            f"País do cliente: {customer['pais']}\n"
            f"Região do cliente: {customer['regiao']}"
        )
        documents.append(
            Document(
                text=text,
                metadata={
                    "sale_id": sale["sale_id"],
                    "vehicle_id": sale["vehicle_id"],
                    "customer_id": sale["customer_id"],
                },
            )
        )
    return documents


def read_index_metadata() -> dict[str, Any] | None:
    if not METADATA_PATH.is_file():
        return None
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Metadados do índice inválidos; o índice será recriado.")
        return None


def persist_index_metadata(data_hash: str) -> None:
    temporary_path = METADATA_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"csv_sha256": data_hash}, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(METADATA_PATH)


@lru_cache(maxsize=1)
def get_sales_index(data_hash: str) -> VectorStoreIndex:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(STORAGE_DIR / "chroma"))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    metadata_matches = read_index_metadata() == {"csv_sha256": data_hash}

    if metadata_matches and collection.count() > 0:
        logger.info("Carregando índice persistido do ChromaDB.")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )

    if collection.count() > 0:
        logger.info("Os CSVs mudaram; recriando a coleção do ChromaDB.")
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("Indexando registros dos três CSVs.")
    index = VectorStoreIndex.from_documents(
        load_sales_documents(),
        storage_context=storage_context,
        show_progress=True,
    )
    persist_index_metadata(data_hash)
    logger.info("Índice persistido com %s registros.", collection.count())
    return index


@cl.on_chat_start
async def on_chat_start() -> None:
    status = cl.Message(content="Preparando os dados de vendas...")
    await status.send()
    try:
        configure_models()
        index = get_sales_index(dataset_hash())
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            text_qa_template=QA_TEMPLATE,
            streaming=True,
        )
        cl.user_session.set("query_engine", query_engine)
        status.content = "Base carregada. Pergunte sobre as vendas de carros."
        logger.info("Base de vendas pronta.")
    except Exception as error:
        logger.exception("Não foi possível preparar o RAG de vendas.")
        status.content = f"Não foi possível preparar a base: {error}"
    await status.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query_engine = cl.user_session.get("query_engine")
    if query_engine is None:
        await cl.Message(
            content="A base ainda não está pronta. Confira os CSVs e as configurações."
        ).send()
        return

    response = await query_engine.aquery(message.content)
    reply = cl.Message(content="")
    await reply.send()
    async for token in response.async_response_gen():
        await reply.stream_token(token)
    await reply.update()

