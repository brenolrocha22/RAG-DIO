import csv
import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path

import chainlit as cl
import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"

load_dotenv(ROOT_DIR / ".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Você é um assistente técnico, direto e útil.
Responda sempre em português do Brasil e use somente informações presentes no contexto.
Se não encontrar a resposta na base, diga isso claramente.

Contexto:
{context_str}

Pergunta e histórico:
{query_str}

Resposta:
"""
QA_PROMPT = PromptTemplate(SYSTEM_PROMPT)


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"A variável {name} não foi configurada. Confira o arquivo .env."
        )
    return value


def setup_settings() -> None:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.llm = OpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        api_key=get_required_env("OPENROUTER_API_KEY"),
        context_window=8192,
        max_tokens=512,
        temperature=0.1,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )


def get_data_files() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in DATA_DIR.rglob("*") if path.is_file())
    if not files:
        sample = DATA_DIR / "exemplo.txt"
        sample.write_text(
            "Este chatbot consulta os arquivos colocados na pasta data.",
            encoding="utf-8",
        )
        files = [sample]
    return files


def get_data_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(DATA_DIR).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_documents(files: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in files:
        if path.suffix.lower() == ".csv":
            with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
                reader = csv.DictReader(csv_file)
                if not reader.fieldnames:
                    raise ValueError(f"O arquivo {path.name} não tem cabeçalho CSV.")
                for row_number, row in enumerate(reader, start=2):
                    fields = [
                        f"{key}: {value}"
                        for key, value in row.items()
                        if key and value is not None and value.strip()
                    ]
                    if fields:
                        documents.append(
                            Document(
                                text="\n".join(fields),
                                metadata={
                                    "file_name": path.name,
                                    "row_number": str(row_number),
                                },
                            )
                        )
        else:
            documents.extend(
                SimpleDirectoryReader(input_files=[str(path)]).load_data()
            )

    if not documents:
        raise ValueError("A pasta data não contém documentos com conteúdo para indexar.")
    return documents


@lru_cache(maxsize=1)
def create_query_engine(data_fingerprint: str):
    files = get_data_files()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        f"dio_rag_{data_fingerprint[:16]}"
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if collection.count() == 0:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            load_documents(files),
            storage_context=storage_context,
        )
    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=Settings.embed_model,
        )

    return index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=QA_PROMPT,
        streaming=True,
    )


def initialize_rag():
    files = get_data_files()
    return create_query_engine(get_data_fingerprint(files))


def build_prompt(history: list[dict[str, str]], user_message: str) -> str:
    recent_history = history[-8:]
    if not recent_history:
        return user_message

    conversation = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in recent_history
    )
    return (
        f"Histórico recente da conversa:\n{conversation}\n\n"
        f"Pergunta atual do usuário:\n{user_message}"
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("history", [])
    status = cl.Message(content="Preparando a base de conhecimento...")
    await status.send()

    try:
        setup_settings()
        cl.user_session.set("query_engine", initialize_rag())
        status.content = "Base de conhecimento carregada. Como posso ajudar?"
        logger.info("Base de conhecimento pronta.")
    except Exception:
        logger.exception("Falha ao inicializar o chatbot.")
        status.content = (
            "Não foi possível preparar o chatbot. Confira a chave da API, "
            "os arquivos da pasta data e o terminal para ver o erro."
        )
    await status.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query_engine = cl.user_session.get("query_engine")
    if query_engine is None:
        await cl.Message(
            content="A base não está pronta. Inicie um novo chat após corrigir o erro."
        ).send()
        return

    history = cl.user_session.get("history") or []
    prompt = build_prompt(history, message.content)
    try:
        response = await query_engine.aquery(prompt)
        reply = cl.Message(content="")
        await reply.send()
        answer = ""
        async for token in response.async_response_gen():
            answer += token
            await reply.stream_token(token)
        await reply.update()
    except Exception:
        logger.exception("Falha ao responder à pergunta.")
        await cl.Message(
            content="Não consegui gerar a resposta. Confira a conexão e o terminal."
        ).send()
        return

    history.extend(
        [
            {"role": "Usuário", "content": message.content},
            {"role": "Assistente", "content": answer},
        ]
    )
    cl.user_session.set("history", history[-12:])
