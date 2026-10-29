from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import chainlit as cl
import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.chroma import ChromaVectorStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from documents import Record, load_sales, sale_records, summary_records  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
COLLECTION_NAME = "vendas_de_carros"
METADATA_PATH = STORAGE_DIR / "index_metadata.json"
SUMMARY_TOP_K = 3
SALES_TOP_K = 5

load_dotenv(BASE_DIR / ".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

QA_TEMPLATE = PromptTemplate(
    """Você é um assistente que consulta uma base de vendas de carros.

O contexto tem dois tipos de registro:
- "Resumo": totais pré-calculados sobre a base completa (por marca, modelo, país, região, ano e rankings).
- "Venda": uma venda individual.

Regras:
- Responda em português do Brasil, de forma direta.
- Use somente as informações do contexto; não invente dados.
- Para totais, contagens, médias e rankings, use apenas os registros "Resumo"; nunca some ou conte vendas individuais para estimar números da base inteira.
- Para perguntas sobre vendas específicas, cite o identificador da venda (por exemplo, VEN-000123).
- Os valores monetários não têm moeda definida na base; não acrescente símbolo de moeda.
- Se a informação não estiver no contexto, diga que não a encontrou na base.

Contexto:
{context_str}

Pergunta:
{query_str}

Resposta:
"""
)


class SummaryAndSalesRetriever(BaseRetriever):
    """Recupera resumos agregados e vendas individuais em consultas separadas.

    Sem essa separação, as ~1.000 vendas individuais dominariam a busca e os
    resumos (necessários para perguntas de totais e rankings) raramente
    apareceriam entre os primeiros resultados.
    """

    def __init__(self, index: VectorStoreIndex) -> None:
        super().__init__()
        self._summaries = index.as_retriever(
            similarity_top_k=SUMMARY_TOP_K, filters=_type_filter("resumo")
        )
        self._sales = index.as_retriever(
            similarity_top_k=SALES_TOP_K, filters=_type_filter("venda")
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._summaries.retrieve(query_bundle) + self._sales.retrieve(query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        summaries = await self._summaries.aretrieve(query_bundle)
        sales = await self._sales.aretrieve(query_bundle)
        return summaries + sales


def _type_filter(tipo: str) -> MetadataFilters:
    return MetadataFilters(filters=[ExactMatchFilter(key="tipo", value=tipo)])


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
        context_window=32768,
        max_tokens=512,
        temperature=0.1,
    )
    Settings.embed_model = FastEmbedEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )


def build_records() -> list[Record]:
    sales = load_sales(DATA_DIR)
    return summary_records(sales) + sale_records(sales)


def records_hash(records: list[Record]) -> str:
    """Muda sempre que os CSVs, o formato dos textos ou o modelo de embeddings mudam."""
    digest = hashlib.sha256(Settings.embed_model.model_name.encode("utf-8"))
    for record in records:
        digest.update(json.dumps([record.text, record.metadata], sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def build_nodes(records: list[Record]) -> list[TextNode]:
    # Cada registro vira um único nó (sem divisão em chunks), para que um
    # resumo nunca seja separado do título que identifica a que ele se refere.
    return [
        TextNode(
            text=record.text,
            metadata=record.metadata,
            excluded_embed_metadata_keys=list(record.metadata),
            excluded_llm_metadata_keys=list(record.metadata),
        )
        for record in records
    ]


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


_index_cache: dict[str, VectorStoreIndex] = {}


def get_sales_index() -> VectorStoreIndex:
    """Reutiliza o índice entre sessões do chat enquanto os registros não mudarem."""
    records = build_records()
    data_hash = records_hash(records)
    if data_hash not in _index_cache:
        _index_cache.clear()
        _index_cache[data_hash] = _load_or_build_index(data_hash, records)
    return _index_cache[data_hash]


def _load_or_build_index(data_hash: str, records: list[Record]) -> VectorStoreIndex:
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
        logger.info("Os dados ou o formato do índice mudaram; recriando a coleção do ChromaDB.")
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("Indexando resumos e vendas dos três CSVs.")
    index = VectorStoreIndex(
        build_nodes(records),
        storage_context=storage_context,
        embed_model=Settings.embed_model,
        show_progress=True,
    )
    persist_index_metadata(data_hash)
    logger.info("Índice persistido com %s registros.", collection.count())
    return index


def build_query_engine() -> RetrieverQueryEngine:
    configure_models()
    index = get_sales_index()
    return RetrieverQueryEngine.from_args(
        SummaryAndSalesRetriever(index),
        llm=Settings.llm,
        text_qa_template=QA_TEMPLATE,
        streaming=True,
    )


def source_label(node: NodeWithScore) -> str:
    metadata = node.node.metadata
    if metadata.get("tipo") == "venda":
        return metadata["sale_id"]
    return node.node.get_content().splitlines()[0]


@cl.on_chat_start
async def on_chat_start() -> None:
    status = cl.Message(content="Preparando os dados de vendas...")
    await status.send()
    try:
        cl.user_session.set("query_engine", build_query_engine())
        status.content = (
            "Base carregada. Pergunte sobre as vendas de carros — por exemplo, "
            "*qual marca mais vendeu?* ou *quanto o Brasil faturou em 2018?*"
        )
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

    sources = list(dict.fromkeys(source_label(node) for node in response.source_nodes))
    if sources:
        await reply.stream_token("\n\n**Fontes consultadas:** " + "; ".join(sources))
    await reply.update()
