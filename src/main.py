import csv
import hashlib
import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import chainlit as cl
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openrouter import OpenRouter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "sales.csv"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_METADATA_PATH = STORAGE_DIR / "index_metadata.json"

load_dotenv(BASE_DIR / ".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

QA_TEMPLATE = PromptTemplate(
    """Você é um assistente especializado em consultar uma base de vendas.

Regras:
- Responda sempre em português do Brasil.
- Use somente informações presentes no contexto.
- Para perguntas sobre preço, informe o preço unitário quando disponível.
- Para quantidades e valores vendidos, use os campos quantidade e valor_total.
- Se o mesmo produto aparecer em mais de uma venda, deixe isso claro.
- Se a informação não estiver no contexto, diga que não a encontrou na base.

Contexto:
{context_str}

Pergunta:
{query_str}

Resposta final:
"""
)


def get_required_env(name: str) -> str:
    """Retorna uma variável de ambiente obrigatória."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"A variável {name} não foi configurada. Confira o arquivo .env."
        )
    return value


def configure_models() -> None:
    Settings.llm = OpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        api_key=get_required_env("OPENROUTER_API_KEY"),
        context_window=8192,
        max_tokens=512,
        temperature=0.1,
    )
    Settings.embed_model = FastEmbedEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )


def get_csv_hash() -> str:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {CSV_PATH}")
    return hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()


def load_sales_documents() -> list[Document]:
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = csv.DictReader(csv_file)
        documents = [
            Document(
                text="\n".join(f"{key}: {value}" for key, value in row.items()),
                metadata={"venda_id": str(row.get("venda_id", ""))},
            )
            for row in rows
        ]

    if not documents:
        raise ValueError(f"O arquivo {CSV_PATH.name} não contém vendas para indexar.")
    return documents


@lru_cache(maxsize=1)
def get_sales_index(csv_hash: str) -> Any:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_METADATA_PATH.is_file():
        try:
            metadata = json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
            if metadata.get("csv_sha256") == csv_hash:
                storage = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
                logger.info("Carregando índice de vendas persistido.")
                return load_index_from_storage(storage)
        except Exception:
            logger.exception("Índice salvo inválido; será reconstruído.")

    logger.info("Criando índice a partir de %s.", CSV_PATH)
    index = VectorStoreIndex.from_documents(load_sales_documents())
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    INDEX_METADATA_PATH.write_text(
        json.dumps(
            {"csv_sha256": csv_hash, "updated_at": time.time()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return index


def build_query(history: list[dict[str, str]], user_message: str) -> str:
    recent_history = history[-8:]
    if not recent_history:
        return user_message

    conversation = "\n".join(
        f"{item['role']}: {item['content']}" for item in recent_history
    )
    return f"Histórico recente:\n{conversation}\n\nPergunta atual:\n{user_message}"


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("history", [])
    status = cl.Message(content="Preparando a base de vendas...")
    await status.send()

    started_at = time.perf_counter()
    try:
        configure_models()
        index = get_sales_index(get_csv_hash())
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            text_qa_template=QA_TEMPLATE,
            streaming=True,
        )
        cl.user_session.set("query_engine", query_engine)
        status.content = "Base de vendas carregada. Faça sua pergunta."
        logger.info("Base pronta em %.2f segundos.", time.perf_counter() - started_at)
    except Exception as error:
        logger.exception("Falha ao preparar a base de vendas.")
        status.content = f"Não foi possível iniciar a base de vendas: {error}"
    await status.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query_engine = cl.user_session.get("query_engine")
    if query_engine is None:
        await cl.Message(
            content="A base não está pronta. Confira o CSV e as configurações do projeto."
        ).send()
        return

    history = cl.user_session.get("history") or []
    query = build_query(history, message.content)
    response = await query_engine.aquery(query)

    reply = cl.Message(content="")
    await reply.send()
    answer = ""
    async for token in response.async_response_gen():
        answer += token
        await reply.stream_token(token)
    await reply.update()

    history.extend(
        [
            {"role": "Usuário", "content": message.content},
            {"role": "Assistente", "content": answer},
        ]
    )
    cl.user_session.set("history", history[-12:])
