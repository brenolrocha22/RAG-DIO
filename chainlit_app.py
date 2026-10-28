import os
from pathlib import Path

import chainlit as cl
import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
load_dotenv(ROOT_DIR / ".env")

SYSTEM_PROMPT = """
Você é um agente técnico, direto, preciso e útil.
Responda sempre em português do Brasil.
Quando não souber a resposta, diga que não sabe e não invente uma resposta.

Use o contexto abaixo para responder.

Contexto:
{context_str}

Pergunta e histórico da conversa:
{query_str}

Resposta:
"""


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"A variável de ambiente {name} não foi configurada. Configure-a no arquivo .env."
        )
    return value


def setup_settings() -> None:
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


def build_prompt(history: list[dict[str, str]], user_message: str) -> str:
    recent_history = history[-8:]
    if not recent_history:
        return user_message

    conversation = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in recent_history
    )
    return (
        f"Histórico recente da conversa:\n{conversation}\n\n"
        f"Usuário: {user_message}\nAssistente:"
    )


def initialize_rag():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if not any(DATA_DIR.iterdir()):
        (DATA_DIR / "exemplo.txt").write_text(
            "O projeto de RAG conecta LlamaIndex, ChromaDB e Chainlit com OpenRouter.",
            encoding="utf-8",
        )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("dio_rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    return index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=PromptTemplate(SYSTEM_PROMPT),
        streaming=True,
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("history", [])
    msg = cl.Message(content="Inicializando a base de conhecimento... Aguarde.")
    await msg.send()

    try:
        setup_settings()
        cl.user_session.set("query_engine", initialize_rag())
        msg.content = "Base de conhecimento carregada. Como posso ajudar?"
    except Exception as error:
        msg.content = f"Erro ao inicializar o RAG: {error}"
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query_engine = cl.user_session.get("query_engine")
    if query_engine is None:
        await cl.Message(
            content="O RAG não foi inicializado. Confira a chave OPENROUTER_API_KEY no arquivo .env."
        ).send()
        return

    history = cl.user_session.get("history") or []
    prompt = build_prompt(history, message.content)
    response = await query_engine.aquery(prompt)

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
