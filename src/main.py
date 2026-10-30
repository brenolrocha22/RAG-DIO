from __future__ import annotations

import asyncio
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
from fastembed.rerank.cross_encoder import TextCrossEncoder
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.agent.workflow import AgentStream, FunctionAgent, ToolCallResult
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.workflow import Context
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.chroma import ChromaVectorStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from database import SCHEMA_DESCRIPTION, build_database, distinct_values, run_select  # noqa: E402
from documents import Record, load_sales, sale_records, summary_records  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
DB_PATH = STORAGE_DIR / "vendas.db"
COLLECTION_NAME = "vendas_de_carros"
METADATA_PATH = STORAGE_DIR / "index_metadata.json"

# A busca vetorial traz candidatos em excesso; o reranker escolhe os melhores.
SUMMARY_CANDIDATES = 10
SALES_CANDIDATES = 20
SUMMARY_TOP_K = 3
SALES_TOP_K = 5
RERANK_MAX_CHARS = 300

DOCUMENT_SEPARATOR = "\n\n---\n\n"
RAG_TOOL = "buscar_documentos"
SQL_TOOL = "consultar_sql"

load_dotenv(BASE_DIR / ".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Você é um analista que responde perguntas sobre uma base de vendas de carros.

Você tem duas ferramentas:
- consultar_sql: executa SQL (SQLite) no banco com todas as vendas. Use para qualquer número
  exato — contagens, somas, médias, rankings, comparações e filtros combinados (marca, modelo,
  país, região, ano, mês) — e para listar vendas que atendem a um critério.
- buscar_documentos: busca semântica em resumos pré-calculados e em vendas individuais. Use para
  visão geral de uma marca, modelo, país ou ano, para perguntas vagas e para exemplos de vendas.

Regras:
- Responda em português do Brasil, de forma direta.
- Use somente dados retornados pelas ferramentas; não invente nada.
- Nunca some ou conte linhas de cabeça: peça o cálculo ao SQL (use ROUND(..., 2) em valores).
- Se uma consulta SQL der erro, corrija-a e tente de novo.
- Cite o identificador das vendas específicas (por exemplo, VEN-000123).
- Os valores monetários não têm moeda definida; não acrescente símbolo de moeda. Formate no
  padrão brasileiro (1.234.567,89).
- Se a informação não existir na base, diga isso.

{schema}

Valores existentes (use exatamente esta grafia em filtros):
- marca: {marcas}
- pais: {paises}
- ano: {anos}
"""


class CrossEncoderReranker:
    """Reordena candidatos lendo a pergunta junto com cada documento (cross-encoder).

    O embedding compara vetores calculados separadamente, o que é rápido mas
    aproximado; o cross-encoder é mais preciso e mais lento, por isso só é
    aplicado aos poucos candidatos que a busca vetorial já trouxe. Só o começo
    de cada documento é lido: os títulos já dizem a que o documento responde, e
    textos longos (como o ranking de 54 modelos) tornariam o reranker lento na CPU.
    """

    def __init__(self, model_name: str) -> None:
        self._model = TextCrossEncoder(model_name, cache_dir=str(MODELS_DIR))

    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
        if not nodes:
            return []
        texts = [node.node.get_content()[:RERANK_MAX_CHARS] for node in nodes]
        scores = self._model.rerank(query, texts)
        for node, score in zip(nodes, scores):
            node.score = float(score)
        return sorted(nodes, key=lambda node: node.score, reverse=True)[:top_n]


class SummaryAndSalesRetriever(BaseRetriever):
    """Recupera resumos agregados e vendas individuais em consultas separadas.

    Sem essa separação, as ~1.000 vendas individuais dominariam a busca e os
    resumos (necessários para perguntas de totais e rankings) raramente
    apareceriam entre os primeiros resultados. Filtros de metadados opcionais
    (marca, país, ano) restringem apenas as vendas.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        reranker: CrossEncoderReranker | None,
        sale_filters: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._reranker = reranker
        summary_k = SUMMARY_CANDIDATES if reranker else SUMMARY_TOP_K
        sales_k = SALES_CANDIDATES if reranker else SALES_TOP_K
        self._summaries = index.as_retriever(
            similarity_top_k=summary_k, filters=_metadata_filter({"tipo": "resumo"})
        )
        self._sales = index.as_retriever(
            similarity_top_k=sales_k,
            filters=_metadata_filter({"tipo": "venda", **(sale_filters or {})}),
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._rerank(
            query_bundle,
            self._summaries.retrieve(query_bundle),
            self._sales.retrieve(query_bundle),
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        summaries = await self._summaries.aretrieve(query_bundle)
        sales = await self._sales.aretrieve(query_bundle)
        return await asyncio.to_thread(self._rerank, query_bundle, summaries, sales)

    def _rerank(
        self,
        query_bundle: QueryBundle,
        summaries: list[NodeWithScore],
        sales: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        if self._reranker is None:
            return summaries[:SUMMARY_TOP_K] + sales[:SALES_TOP_K]
        query = query_bundle.query_str
        return self._reranker.rerank(query, summaries, SUMMARY_TOP_K) + self._reranker.rerank(
            query, sales, SALES_TOP_K
        )


def _metadata_filter(values: dict[str, str]) -> MetadataFilters:
    return MetadataFilters(
        filters=[ExactMatchFilter(key=key, value=value) for key, value in values.items()]
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
        model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
        api_key=required_env("OPENROUTER_API_KEY"),
        context_window=128000,
        max_tokens=1024,
        temperature=0.1,
        is_function_calling_model=True,
    )
    Settings.embed_model = FastEmbedEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        cache_dir=str(MODELS_DIR),
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


_reranker_cache: dict[str, CrossEncoderReranker] = {}


def get_reranker() -> CrossEncoderReranker | None:
    """Opcional: só é carregado se RERANK_MODEL estiver definido.

    Ele não tem custo de API, mas soma alguns segundos por pergunta na CPU e
    centenas de MB de memória; por isso fica desligado por padrão.
    """
    model_name = os.getenv("RERANK_MODEL", "")
    if not model_name:
        return None
    if model_name not in _reranker_cache:
        logger.info("Carregando o reranker %s.", model_name)
        _reranker_cache[model_name] = CrossEncoderReranker(model_name)
    return _reranker_cache[model_name]


def format_documents(nodes: list[NodeWithScore]) -> str:
    if not nodes:
        return "Nenhum documento encontrado com esses filtros."
    return DOCUMENT_SEPARATOR.join(node.node.get_content() for node in nodes)


def build_agent() -> FunctionAgent:
    configure_models()
    build_database(DATA_DIR, DB_PATH)
    index = get_sales_index()
    reranker = get_reranker()

    async def buscar_documentos(
        pergunta: str,
        marca: str | None = None,
        pais: str | None = None,
        ano: int | None = None,
    ) -> str:
        """Busca resumos pré-calculados e vendas individuais parecidos com a pergunta.

        Os filtros opcionais marca, pais e ano restringem as vendas individuais
        retornadas; use a grafia exata da lista de valores existentes.
        """
        sale_filters = {
            key: str(value)
            for key, value in {"marca": marca, "pais": pais, "ano": ano}.items()
            if value
        }
        retriever = SummaryAndSalesRetriever(index, reranker, sale_filters)
        return format_documents(await retriever.aretrieve(pergunta))

    async def consultar_sql(sql: str) -> str:
        """Executa uma consulta SELECT (SQLite) somente leitura no banco de vendas.

        Retorna no máximo 50 linhas; agregue no SQL em vez de listar muitas linhas.
        """
        return await asyncio.to_thread(run_select, DB_PATH, sql)

    system_prompt = SYSTEM_PROMPT.format(
        schema=SCHEMA_DESCRIPTION,
        marcas=", ".join(distinct_values(DB_PATH, "marca")),
        paises=", ".join(distinct_values(DB_PATH, "pais")),
        anos=", ".join(distinct_values(DB_PATH, "ano")),
    )
    return FunctionAgent(
        tools=[
            FunctionTool.from_defaults(async_fn=buscar_documentos, name=RAG_TOOL),
            FunctionTool.from_defaults(async_fn=consultar_sql, name=SQL_TOOL),
        ],
        llm=Settings.llm,
        system_prompt=system_prompt,
    )


def source_labels(tool_name: str, output: str) -> list[str]:
    if tool_name == SQL_TOOL:
        return ["banco SQL"]
    # A primeira linha de cada documento identifica a venda ou o resumo.
    return [document.splitlines()[0] for document in output.split(DOCUMENT_SEPARATOR) if document]


@cl.on_chat_start
async def on_chat_start() -> None:
    status = cl.Message(content="Preparando os dados de vendas...")
    await status.send()
    try:
        agent = await asyncio.to_thread(build_agent)
        cl.user_session.set("agent", agent)
        cl.user_session.set("context", Context(agent))
        status.content = (
            "Base carregada. Pergunte sobre as vendas de carros — por exemplo, "
            "*qual marca mais vendeu?*, *quanto o Brasil faturou em 2018?* ou "
            "*quantos Honda Civic foram vendidos na França em 2016?*"
        )
        logger.info("Base de vendas pronta.")
    except Exception as error:
        logger.exception("Não foi possível preparar o RAG de vendas.")
        status.content = f"Não foi possível preparar a base: {error}"
    await status.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    agent = cl.user_session.get("agent")
    if agent is None:
        await cl.Message(
            content="A base ainda não está pronta. Confira os CSVs e as configurações."
        ).send()
        return

    handler = agent.run(user_msg=message.content, ctx=cl.user_session.get("context"))
    reply = cl.Message(content="")
    sources: list[str] = []
    try:
        async for event in handler.stream_events():
            if isinstance(event, AgentStream) and event.delta:
                await reply.stream_token(event.delta)
            elif isinstance(event, ToolCallResult):
                # Fecha o texto já escrito para que a etapa da ferramenta apareça
                # na ordem em que aconteceu.
                if reply.content:
                    await reply.update()
                    reply = cl.Message(content="")
                output = str(event.tool_output)
                async with cl.Step(name=event.tool_name, type="tool") as step:
                    step.input = event.tool_kwargs
                    step.output = output
                sources += source_labels(event.tool_name, output)
        await handler
    except Exception as error:
        logger.exception("Falha ao responder a pergunta.")
        await reply.stream_token(f"\n\nNão foi possível concluir a resposta: {error}")

    unique_sources = list(dict.fromkeys(sources))
    if unique_sources:
        await reply.stream_token("\n\n**Fontes consultadas:** " + "; ".join(unique_sources))
    await reply.update()
