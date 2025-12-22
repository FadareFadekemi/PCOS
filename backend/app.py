import os
import json
import uuid
import logging
from typing import List, AsyncGenerator, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import openai
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from supabase import create_client, Client
from tavily import TavilyClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | PCOS-RAG | %(message)s",
)
logger = logging.getLogger("PCOS-RAG")


class Config:
    OPENAI_GEN_MODEL = "gpt-4o-mini"
    OPENAI_EMBED_MODEL = "text-embedding-3-small"

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    SUPABASE_TABLE = "documents"
    CHAT_SESSIONS_TABLE = "chat_sessions"
    CHAT_MESSAGES_TABLE = "chat_messages"

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    DATA_DIR = "data"
    DATA_FILE = "pcos_question_and_answer_pairs_data.json"


class ChatRequest(BaseModel):
    message: str
    user_id: str
    app: Optional[str] = "CystaCare"
    history: Optional[List[Dict]] = None


class ChatMemorySaveRequest(BaseModel):
    user_id: str
    messages: List[Dict]


class PCOSRAG:
    def __init__(self):
        self.openai = openai.AsyncOpenAI()
        self.embeddings = OpenAIEmbeddings(model=Config.OPENAI_EMBED_MODEL)
        self.supabase: Client = create_client(
            Config.SUPABASE_URL,
            Config.SUPABASE_SERVICE_KEY
        )
        self.tavily = TavilyClient(api_key=Config.TAVILY_API_KEY)

    async def ingest_qa_data(self):
        logger.info("Checking if knowledge base already exists")
        existing = (
            self.supabase
            .table(Config.SUPABASE_TABLE)
            .select("id")
            .limit(1)
            .execute()
        )
        if existing.data:
            logger.info("Ingestion skipped: KB already exists")
            return

        path = os.path.join(Config.DATA_DIR, Config.DATA_FILE)
        if not os.path.exists(path):
            logger.warning("QA data file not found")
            return

        with open(path, "r", encoding="utf-8") as f:
            qa = json.load(f)

        docs = [
            Document(
                page_content=f"Q: {i['question']}\nA: {i['answer']}",
                metadata={"source": "internal_kb"}
            )
            for i in qa
        ]

        logger.info(f"Ingesting {len(docs)} documents")
        for doc in docs:
            emb = await self.embeddings.aembed_query(doc.page_content)
            self.supabase.table(Config.SUPABASE_TABLE).insert({
                "content": doc.page_content,
                "embedding": emb,
                "metadata": doc.metadata
            }).execute()
        logger.info("Knowledge base ingestion complete")

    async def classify_intent(self, text: str) -> str:
        logger.info("Classifying intent using LLM")
        resp = await self.openai.chat.completions.create(
            model=Config.OPENAI_GEN_MODEL,
            messages=[
                {"role": "system",
                 "content": (
                     "Classify the message into:\n"
                     "- small_talk: casual greetings or chit-chat\n"
                     "- pcos: any question related to PCOS including symptoms, diet, lifestyle, supplements, "
                     "medications, hormones, fertility, menstrual cycles, research, and treatments\n"
                     "- other: unrelated questions\n"
                     "Return ONLY the label."
                 )},
                {"role": "user", "content": text}
            ]
        )
        intent = resp.choices[0].message.content.strip().lower()
        logger.info(f"Initial intent: {intent}")

        # Secondary PCOS relevance check if initially classified as 'other'
        if intent == "other":
            logger.info("Running secondary PCOS relevance check")
            check = await self.openai.chat.completions.create(
                model=Config.OPENAI_GEN_MODEL,
                messages=[
                    {"role": "system",
                     "content": "Answer ONLY YES or NO. "
                                "Does this question relate to PCOS or hormonal health?"},
                    {"role": "user", "content": text}
                ]
            )
            verdict = check.choices[0].message.content.strip().upper()
            if verdict == "YES":
                logger.info("Reclassified as PCOS based on secondary check")
                intent = "pcos"

        logger.info(f"Final intent classified as: {intent}")
        return intent

    async def retrieve_from_db(self, query: str) -> List[Document]:
        logger.info("Attempting retrieval from vector database")
        emb = await self.embeddings.aembed_query(query)
        resp = self.supabase.rpc(
            "match_documents",
            {"query_embedding": emb, "match_threshold": 0.4, "match_count": 3}
        ).execute()
        docs = [Document(page_content=row["content"]) for row in (resp.data or [])]
        logger.info(f"DB retrieval returned {len(docs)} documents")
        return docs

    async def db_answers_question(self, query: str, docs: List[Document]) -> bool:
        if not docs:
            return False
        logger.info("Checking if DB documents actually answer the question")
        ctx = "\n\n".join(d.page_content for d in docs)
        resp = await self.openai.chat.completions.create(
            model=Config.OPENAI_GEN_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a strict evaluator.\nAnswer ONLY YES or NO.\nDo the provided documents fully answer the user question?"},
                {"role": "user", "content": f"DOCUMENTS:\n{ctx}\n\nQUESTION:\n{query}"}
            ]
        )
        decision = resp.choices[0].message.content.upper().strip()
        logger.info(f"Answerability verdict: {decision}")
        return decision == "YES"

    async def search_web(self, query: str) -> Dict:
        logger.info("Falling back to Tavily web search")
        results = self.tavily.search(
            query=f"PCOS {query}",
            max_results=3,
            search_depth="advanced"
        )
        return {
            "content": "\n".join(r["content"] for r in results["results"]),
            "sources": [r["url"] for r in results["results"]]
        }

    async def stream_answer(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:
        stream = await self.openai.chat.completions.create(
            model=Config.OPENAI_GEN_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def chat(self, message: str) -> AsyncGenerator[str, None]:
        logger.info("---- NEW CHAT REQUEST ----")
        logger.info(f"User message: {message}")
        intent = await self.classify_intent(message)

        if intent == "small_talk":
            yield "Hello! I'm your PCOS assistant. How can I help you today?"
            return

        if intent != "pcos":
            yield "I specialize in PCOS and hormonal health questions."
            return

        docs = await self.retrieve_from_db(message)
        if await self.db_answers_question(message, docs):
            logger.info("Answering from INTERNAL DATABASE")
            ctx = "\n\n".join(d.page_content for d in docs)
            async for chunk in self.stream_answer(
                    system_prompt="Answer ONLY using the provided documents.",
                    user_prompt=f"{ctx}\n\nQuestion: {message}"):
                yield chunk
            return

        web = await self.search_web(message)
        if not web["content"].strip():
            yield "I couldn't find reliable information for that question."
            return

        logger.info("Answering from TAVILY WEB SEARCH")
        async for chunk in self.stream_answer(
                system_prompt="Answer using ONLY the provided web research. Cite sources at the end.",
                user_prompt=f"RESEARCH:\n{web['content']}\n\nQUESTION:\n{message}"):
            yield chunk
        if web["sources"]:
            yield "\n\nSources:\n"
            for src in web["sources"]:
                yield f"- {src}\n"

    
    def get_or_create_session(self, user_id: str) -> str:
        logger.info(f"Fetching session for user_id={user_id}")
        resp = self.supabase.table(Config.CHAT_SESSIONS_TABLE).select("*").eq("user_id", user_id).limit(1).execute()
        if resp.data:
            session_id = resp.data[0]["id"]
            logger.info(f"Found existing session: {session_id}")
            return session_id
        session_id = str(uuid.uuid4())
        logger.info(f"Creating new session: {session_id}")
        self.supabase.table(Config.CHAT_SESSIONS_TABLE).insert({"id": session_id, "user_id": user_id}).execute()
        return session_id

    def save_messages(self, user_id: str, messages: List[Dict]):
        session_id = self.get_or_create_session(user_id)
        for msg in messages:
            self.supabase.table(Config.CHAT_MESSAGES_TABLE).insert({
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "role": msg.get("role"),
                "content": msg.get("content"),
                "source": msg.get("source", "user")
            }).execute()
        logger.info(f"Saved {len(messages)} messages for session {session_id}")

    def fetch_messages(self, user_id: str, limit: int = 20) -> List[Dict]:
        session_id = self.get_or_create_session(user_id)
        resp = self.supabase.table(Config.CHAT_MESSAGES_TABLE)\
            .select("*")\
            .eq("session_id", session_id)\
            .order("created_at", desc=True)\
            .limit(limit).execute()
        messages = [{"role": r["role"], "content": r["content"], "source": r.get("source", "user")}
                    for r in reversed(resp.data or [])]
        logger.info(f"Fetched {len(messages)} messages for session {session_id}")
        return messages

rag = PCOSRAG()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await rag.ingest_qa_data()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/chat-stream")
async def chat_endpoint(req: ChatRequest):
    async def generator():
        rag.save_messages(req.user_id, [{"role": "user", "content": req.message, "source": "user"}])
        full_response = ""
        async for chunk in rag.chat(req.message):
            full_response += chunk
            yield chunk
        rag.save_messages(req.user_id, [{"role": "assistant", "content": full_response, "source": "assistant"}])
    return StreamingResponse(generator(), media_type="text/plain")


@app.get("/chat-memory")
def get_chat_memory(user_id: str = Query(...), limit: int = 20):
    messages = rag.fetch_messages(user_id, limit=limit)
    return messages


@app.post("/chat-memory")
def save_chat_memory(req: ChatMemorySaveRequest):
    rag.save_messages(req.user_id, req.messages)
    return {"status": "ok", "saved": len(req.messages)}


@app.get("/")
def root():
    return {"status": "PCOS RAG running"}
