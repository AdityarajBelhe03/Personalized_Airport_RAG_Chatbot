# Complete Changi Airport Chatbot – Production Ready (concise edition)
import os
import re
import asyncio
import logging
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from collections import defaultdict

# ───────────────────────────  logging  ────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────────────────────────  data types  ─────────────────────────
class QueryType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    MULTI_DOMAIN = "multi_domain"
    LOCATION_SPECIFIC = "location_specific"
    PRICE_FILTERED = "price_filtered"


@dataclass
class RetrievalResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    category: str
    source: str


@dataclass
class QueryAnalysis:
    query_type: QueryType
    categories: List[str]
    location_filters: List[str]
    price_filters: Dict[str, Any]
    dietary_requirements: List[str]
    intent: str
    keywords: List[str]

# ──────────────────────  Query analysis  ──────────────────────────
class AdvancedQueryAnalyzer:
    """Light-weight query analyser"""

    def __init__(self):
        self.categories = ['shops-retail', 'dining', 'services',
                           'transportation', 'attractions']
        self.locations = ['terminal 1', 'terminal 2', 'terminal 3',
                          'terminal 4', 'jewel', 'transit area']
        self.dietary_keywords = ['halal', 'vegetarian', 'vegan',
                                 'kosher', 'gluten-free']

    def analyze_query(self, query: str) -> QueryAnalysis:
        ql = query.lower()

        categories = [c for c in self.categories if any(k in ql
                       for k in self._category_kw(c))]
        locations = [l for l in self.locations if l in ql]
        dietary = [d for d in self.dietary_keywords if d in ql]
        price_filters = self._extract_price_filters(ql)

        if len(categories) > 1:
            qtype = QueryType.MULTI_DOMAIN
        elif locations and price_filters:
            qtype = QueryType.COMPLEX
        elif locations:
            qtype = QueryType.LOCATION_SPECIFIC
        elif price_filters:
            qtype = QueryType.PRICE_FILTERED
        else:
            qtype = QueryType.SIMPLE

        return QueryAnalysis(
            query_type=qtype,
            categories=categories or self.categories,
            location_filters=locations,
            price_filters=price_filters,
            dietary_requirements=dietary,
            intent='general',
            keywords=self._extract_keywords(query)
        )

    # helpers
    def _category_kw(self, category):
        m = {
            'dining': ['restaurant', 'food', 'eat', 'cafe', 'bar', 'halal'],
            'shops-retail': ['shop', 'store', 'retail', 'shopping'],
            'services': ['service', 'lounge', 'wifi'],
            'transportation': ['taxi', 'bus', 'mrt', 'train'],
            'attractions': ['attraction', 'see', 'jewel', 'waterfall']
        }
        return m.get(category, [])

    def _extract_price_filters(self, q):
        for pat in [r'under \$?(\d+)', r'below \$?(\d+)']:
            m = re.search(pat, q)
            if m:
                return {'max_price': int(m.group(1))}
        return {}

    def _extract_keywords(self, q):
        stop = {'the', 'and', 'or', 'but', 'for', 'with', 'a', 'an', 'to'}
        return [w for w in re.findall(r'\b\w+\b', q.lower())
                if w not in stop and len(w) > 2]

# ───────────────────  Retriever (Pinecone)  ───────────────────────
class MultiDomainRetriever:
    def __init__(self, index, embedder, analyzer):
        self.index = index
        self.embedder = embedder
        self.analyzer = analyzer

    async def retrieve(self, query, top_k=6) -> List[RetrievalResult]:
        analysis = self.analyzer.analyze_query(query)
        emb = self.embedder.encode(query, normalize_embeddings=True).tolist()
        namespace = analysis.categories[0]

        try:
            resp = self.index.query(
                vector=emb, top_k=top_k, namespace=namespace,
                include_metadata=True
            )
            return [RetrievalResult(
                content=m['metadata'].get('text', ''),
                metadata=m['metadata'],
                score=m['score'],
                category=namespace,
                source=m['metadata'].get('source', '')
            ) for m in resp.get('matches', [])]
        except Exception as e:
            logger.warning(f"Pinecone query failed: {e}")
            return []

# ───────────────────────  Chatbot Core  ───────────────────────────
class ChangiAirportChatbot:
    def __init__(self, pinecone_api_key, index_name, groq_api_key):
        # Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)

        # Embeddings
        self.embedder = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2')

        # Components
        self.analyzer = AdvancedQueryAnalyzer()
        self.retriever = MultiDomainRetriever(
            self.index, self.embedder, self.analyzer)

        # LLM
        self.client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0
        )
        self.model_name = "llama-3.1-8b-instant"

        # Memory (simple)
        self.history = []
        self.max_history = 6

    # ───────────── prompt ─────────────
    def _create_system_prompt(self, analysis: QueryAnalysis) -> str:
        """Create dynamic system prompt based on query analysis"""

        base_prompt = """You are Changi, an expert and friendly AI assistant for Changi Airport and Jewel Changi Airport in Singapore. You're known for being incredibly helpful, knowledgeable, and having a warm personality with just the right touch of playfulness while maintaining professionalism.

CORE RESPONSIBILITIES:
- Provide accurate, helpful information about Changi Airport and Jewel facilities
- Use ONLY the provided context to answer questions
- Be specific with details like operating hours, locations, terminal numbers, and contact information
- If information isn't available in the context, clearly state this limitation
- Do NOT include source citations in your response text

PERSONALITY TRAITS:
- Warm and welcoming, like a knowledgeable local friend
- Professional yet approachable
- Occasionally playful but never inappropriate
- Proactive in offering helpful suggestions
- Empathetic to traveler needs and stress

RESPONSE STRUCTURE:
1. Direct answer to the user's question
2. Relevant additional details that might be helpful
3. NO follow-up questions or suggestions

FORMATTING REQUIREMENTS:
- Keep responses concise (maximum 4-5 sentences total)
- When listing multiple items, ALWAYS use numbered lists (1., 2., 3.)
- Use **bold** for facility/store names and *italics* for locations
- Avoid long paragraphs - break information into structured points
- Each list item should be one clear, concise line
- Never write more than 2 sentences in a row without using a list format
- Do NOT add follow-up questions or suggestions at the end

SPECIAL INSTRUCTIONS:"""

        # Add query-specific instructions
        if analysis.query_type == QueryType.COMPLEX:
            base_prompt += """
- You're handling a complex query with multiple requirements
- Break down your response to address each requirement clearly
- Prioritize the most relevant matches first
- If some requirements can't be fully met, explain what's available"""

        if analysis.categories:
            categories_str = ", ".join(analysis.categories)
            base_prompt += f"""
- Focus on information from these categories: {categories_str}
- Provide comprehensive coverage across the relevant areas"""

        if analysis.location_filters:
            locations_str = ", ".join(analysis.location_filters)
            base_prompt += f"""
- Pay special attention to locations: {locations_str}
- Include specific terminal or area information when relevant"""

        if analysis.dietary_requirements:
            dietary_str = ", ".join(analysis.dietary_requirements)
            base_prompt += f"""
- The user has specific dietary requirements: {dietary_str}
- Highlight relevant dietary information clearly
- Mention certifications or dietary compliance when available"""

        base_prompt += """

Remember: You're not just providing information, you're helping create a positive airport experience. Be genuinely helpful and make travelers feel welcomed and well-informed about their time at Changi Airport. Always use structured formatting with numbered lists when presenting multiple options."""

        return base_prompt

    # ───────────── chat ─────────────
    async def chat(self, user_msg: str) -> Dict[str, Any]:
        analysis = self.analyzer.analyze_query(user_msg)

        docs = await self.retriever.retrieve(user_msg, top_k=6)
        context = "\n---\n".join(d.content[:600] for d in docs[:3])  # brevity

        messages = [
            {"role": "system", "content": self._create_system_prompt(analysis)},
            *self.history[-self.max_history:],  # keep last few turns
            {"role": "user", "content": user_msg if not context else
             f"Context:\n{context}\n\nUser: {user_msg}"}
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.5,
                max_tokens=300,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = ("Sorry, I'm unable to fetch information right now. "
                      "Please check the official Changi Airport website.")

        # post-process: drop citations or questions
        answer = re.sub(r'\(.*?source.*?\)', '', answer, flags=re.I)
        answer = re.sub(r'💡.*', '', answer).strip()
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        # Remove any trailing questions
        answer = re.sub(r'\?[^?]*$', '', answer).strip()

        # update memory (no large history needed)
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-2*self.max_history:]

        return {
            "response": answer,
            "sources": [],                 # not shown to user
            "query_analysis": analysis.__dict__,
            "retrieved_count": len(docs)
        }

# ───────────────────────────  FastAPI  ────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    session_id: str
    query_analysis: Dict[str, Any]
    retrieved_count: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


chatbot: Optional[ChangiAirportChatbot] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    try:
        pc_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        idx_name = os.getenv("INDEX_NAME", "airportchatbot")
        if not (pc_key and groq_key):
            logger.warning("Missing API keys – chatbot disabled")
            yield
            return

        chatbot = ChangiAirportChatbot(pc_key, idx_name, groq_key)
        logger.info("Chatbot ready")
        yield
    finally:
        logger.info("App shutdown")


app = FastAPI(lifespan=lifespan,
              title="Changi Airport Chatbot API",
              version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────  endpoints  ────────────────────────────
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if chatbot else "starting",
        timestamp=datetime.utcnow().isoformat(),
        version="2.0"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not ready")

    result = await chatbot.chat(req.message)
    return ChatResponse(
        response=result["response"],
        sources=result["sources"],
        session_id=req.session_id or "default",
        query_analysis=result["query_analysis"],
        retrieved_count=result["retrieved_count"],
        timestamp=datetime.utcnow().isoformat()
    )


# optional simple search
@app.post("/search")
async def search(req: ChatRequest):
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not ready")
    docs = await chatbot.retriever.retrieve(req.message, top_k=10)
    return {"hits": len(docs)}

# ──────────────────────────  run local  ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), log_level="info")
