# Complete Changi Airport Chatbot - Production Ready
import os
import json
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import re
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Core dependencies - FIXED PINECONE IMPORT
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from collections import defaultdict
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# CHATBOT CLASSES
# ===============================================

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

class AdvancedQueryAnalyzer:
    """Sophisticated query analysis for multi-domain retrieval"""

    def __init__(self):
        self.categories = ['shops-retail', 'dining', 'services', 'transportation', 'attractions']
        self.locations = ['terminal 1', 'terminal 2', 'terminal 3', 'terminal 4', 'jewel', 'transit area']
        self.dietary_keywords = ['halal', 'vegetarian', 'vegan', 'kosher', 'gluten-free']
        self.price_keywords = ['cheap', 'expensive', 'budget', 'affordable', 'luxury', 'premium']

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine retrieval strategy"""
        query_lower = query.lower()

        # Detect categories mentioned
        categories = [cat for cat in self.categories if any(keyword in query_lower
                     for keyword in self._get_category_keywords(cat))]

        # Detect locations
        locations = [loc for loc in self.locations if loc in query_lower]

        # Extract price filters
        price_filters = self._extract_price_filters(query_lower)

        # Detect dietary requirements
        dietary = [diet for diet in self.dietary_keywords if diet in query_lower]

        # Determine query type
        query_type = self._determine_query_type(query_lower, categories, locations, price_filters)

        # Extract intent and keywords
        intent = self._extract_intent(query_lower)
        keywords = self._extract_keywords(query)

        return QueryAnalysis(
            query_type=query_type,
            categories=categories if categories else self.categories,
            location_filters=locations,
            price_filters=price_filters,
            dietary_requirements=dietary,
            intent=intent,
            keywords=keywords
        )

    def _get_category_keywords(self, category: str) -> List[str]:
        keyword_map = {
            'dining': ['restaurant', 'food', 'eat', 'dining', 'cafe', 'bar', 'meal', 'cuisine', 'halal', 'vegetarian'],
            'shops-retail': ['shop', 'store', 'retail', 'buy', 'purchase', 'souvenir', 'duty-free', 'brand', 'shopping'],
            'services': ['service', 'help', 'assistance', 'wifi', 'lounge', 'pharmacy', 'bank', 'atm'],
            'transportation': ['transport', 'taxi', 'bus', 'mrt', 'train', 'car', 'shuttle', 'uber', 'grab'],
            'attractions': ['attraction', 'visit', 'see', 'experience', 'activity', 'entertainment', 'jewel', 'waterfall', 'rain vortex', 'canopy']
        }
        return keyword_map.get(category, [])

    def _extract_price_filters(self, query: str) -> Dict[str, Any]:
        price_filters = {}

        # Extract specific price ranges
        price_patterns = [
            r'under \$?(\d+)',
            r'below \$?(\d+)',
            r'less than \$?(\d+)',
            r'maximum \$?(\d+)',
            r'max \$?(\d+)'
        ]

        for pattern in price_patterns:
            match = re.search(pattern, query)
            if match:
                price_filters['max_price'] = int(match.group(1))
                break

        # Price range detection
        range_pattern = r'\$?(\d+)\s*-\s*\$?(\d+)'
        range_match = re.search(range_pattern, query)
        if range_match:
            price_filters['min_price'] = int(range_match.group(1))
            price_filters['max_price'] = int(range_match.group(2))

        return price_filters

    def _determine_query_type(self, query: str, categories: List[str], locations: List[str], price_filters: Dict) -> QueryType:
        if len(categories) > 1:
            return QueryType.MULTI_DOMAIN
        elif locations and price_filters:
            return QueryType.COMPLEX
        elif locations:
            return QueryType.LOCATION_SPECIFIC
        elif price_filters:
            return QueryType.PRICE_FILTERED
        else:
            return QueryType.SIMPLE

    def _extract_intent(self, query: str) -> str:
        intent_patterns = {
            'find': ['find', 'locate', 'where', 'search'],
            'recommend': ['recommend', 'suggest', 'best', 'good', 'popular'],
            'compare': ['compare', 'difference', 'versus', 'vs', 'better'],
            'navigate': ['how to get', 'directions', 'way to', 'route'],
            'information': ['what is', 'tell me about', 'information', 'details', 'hours', 'operating']
        }

        for intent, keywords in intent_patterns.items():
            if any(keyword in query for keyword in keywords):
                return intent
        return 'general'

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]

class MultiDomainRetriever:
    """Advanced retriever with multi-namespace and filtering capabilities"""

    def __init__(self, pinecone_index, embedding_model, query_analyzer):
        self.index = pinecone_index
        self.embedding_model = embedding_model
        self.query_analyzer = query_analyzer

    async def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve relevant documents with advanced filtering"""

        # Analyze query
        analysis = self.query_analyzer.analyze_query(query)
        logger.info(f"Query analysis: {analysis}")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()

        # Execute retrieval strategy based on query type
        if analysis.query_type == QueryType.MULTI_DOMAIN:
            return await self._multi_domain_retrieval(query_embedding, analysis, top_k)
        elif analysis.query_type == QueryType.COMPLEX:
            return await self._complex_retrieval(query_embedding, analysis, top_k)
        else:
            return await self._standard_retrieval(query_embedding, analysis, top_k)

    async def _multi_domain_retrieval(self, query_embedding: List[float], analysis: QueryAnalysis, top_k: int) -> List[RetrievalResult]:
        """Retrieve from multiple namespaces and merge results"""
        all_results = []

        if len(analysis.categories) > top_k:
            k_per_category = max(2, top_k // len(analysis.categories))
        else:
            k_per_category = top_k

        for category in analysis.categories:
            try:
                filter_dict = self._build_filter(analysis, category)

                response = self.index.query(
                    vector=query_embedding,
                    top_k=k_per_category,
                    namespace=category,
                    filter=filter_dict if filter_dict else None,
                    include_metadata=True
                )

                for match in response['matches']:
                    result = RetrievalResult(
                        content=match['metadata'].get('text', ''),
                        metadata=match['metadata'],
                        score=match['score'],
                        category=category,
                        source=match['metadata'].get('source', '')
                    )
                    all_results.append(result)

            except Exception as e:
                logger.error(f"Error retrieving from {category}: {e}")

        # Apply additional filtering and boost scores
        filtered_results = self._apply_post_retrieval_filters(all_results, analysis)

        # Sort by relevance and return top results
        return sorted(filtered_results, key=lambda x: x.score, reverse=True)[:top_k]

    async def _complex_retrieval(self, query_embedding: List[float], analysis: QueryAnalysis, top_k: int) -> List[RetrievalResult]:
        """Handle complex queries with multiple constraints"""
        results = []

        for category in analysis.categories:
            try:
                filter_dict = self._build_filter(analysis, category)

                response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    namespace=category,
                    filter=filter_dict if filter_dict else None,
                    include_metadata=True
                )

                for match in response['matches']:
                    result = RetrievalResult(
                        content=match['metadata'].get('text', ''),
                        metadata=match['metadata'],
                        score=match['score'],
                        category=category,
                        source=match['metadata'].get('source', '')
                    )
                    results.append(result)

            except Exception as e:
                logger.error(f"Error in complex retrieval for {category}: {e}")

        filtered_results = self._apply_post_retrieval_filters(results, analysis)
        return sorted(filtered_results, key=lambda x: x.score, reverse=True)[:top_k]

    async def _standard_retrieval(self, query_embedding: List[float], analysis: QueryAnalysis, top_k: int) -> List[RetrievalResult]:
        """Standard single-domain retrieval"""
        results = []
        category = analysis.categories[0] if analysis.categories else 'dining'

        try:
            filter_dict = self._build_filter(analysis, category)

            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=category,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )

            for match in response['matches']:
                result = RetrievalResult(
                    content=match['metadata'].get('text', ''),
                    metadata=match['metadata'],
                    score=match['score'],
                    category=category,
                    source=match['metadata'].get('source', '')
                )
                results.append(result)

        except Exception as e:
            logger.error(f"Error in standard retrieval: {e}")

        return results

    def _build_filter(self, analysis: QueryAnalysis, category: str) -> Dict:
        """Build Pinecone filter based on query analysis"""
        filter_dict = {}

        # Location filtering
        if analysis.location_filters:
            # Use text search for location matching
            filter_conditions = []
            for location in analysis.location_filters:
                filter_conditions.append({"text": {"$contains": location}})
            if len(filter_conditions) == 1:
                filter_dict.update(filter_conditions[0])
            else:
                filter_dict["$or"] = filter_conditions

        # Dietary requirements for dining category
        if analysis.dietary_requirements and category == 'dining':
            dietary_conditions = []
            for dietary in analysis.dietary_requirements:
                dietary_conditions.append({"text": {"$contains": dietary}})
            if dietary_conditions:
                if "$or" in filter_dict:
                    filter_dict = {"$and": [filter_dict, {"$or": dietary_conditions}]}
                else:
                    filter_dict["$or"] = dietary_conditions

        return filter_dict

    def _apply_post_retrieval_filters(self, results: List[RetrievalResult], analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Apply additional filtering after retrieval"""
        filtered_results = []

        for result in results:
            # Price filtering
            if analysis.price_filters and 'max_price' in analysis.price_filters:
                price_text = result.content.lower()
                price_matches = re.findall(r'\$(\d+)', price_text)
                if price_matches:
                    prices = [int(p) for p in price_matches]
                    min_price = min(prices)
                    if min_price > analysis.price_filters['max_price']:
                        continue

            # Keyword relevance boost
            content_lower = result.content.lower()
            keyword_matches = sum(1 for keyword in analysis.keywords if keyword in content_lower)
            if keyword_matches > 0:
                result.score *= (1 + keyword_matches * 0.1)

            filtered_results.append(result)

        return filtered_results

class ChangiAirportChatbot:
    """Main chatbot class with advanced conversational capabilities"""

    def __init__(self, pinecone_api_key: str, index_name: str, groq_api_key: str):
        # Initialize Pinecone - FIXED FOR VERSION 3.0.3
        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(index_name)
            logger.info("Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

        # Initialize embedding model with caching
        logger.info("Loading embedding model...")
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Initialize components
        self.query_analyzer = AdvancedQueryAnalyzer()
        self.retriever = MultiDomainRetriever(self.index, self.embedding_model, self.query_analyzer)

        # Initialize conversation memory
        self.conversation_history = []
        self.max_history = 10

        # Initialize OpenAI client with Groq
        try:
            self.client = OpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=30.0
            )
            self.model_name = "mixtral-8x7b-32768"
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

        # Conversation suggestions
        self.suggestions = {
            'dining': [
                "Would you like to know about halal dining options?",
                "Interested in restaurants with great views?",
                "Should I suggest some budget-friendly food courts?"
            ],
            'shopping': [
                "Looking for duty-free shopping deals?",
                "Want to know about luxury brand stores?",
                "Interested in local Singapore souvenirs?"
            ],
            'services': [
                "Need information about lounges or rest areas?",
                "Want to know about WiFi and charging stations?",
                "Looking for banking or currency exchange services?"
            ],
            'transportation': [
                "Need help with ground transportation options?",
                "Want to know about MRT connections?",
                "Looking for parking information?"
            ],
            'attractions': [
                "Interested in Jewel's Rain Vortex show times?",
                "Want to explore Canopy Park activities?",
                "Looking for family-friendly attractions?"
            ]
        }

    def _create_system_prompt(self, analysis: QueryAnalysis) -> str:
        """Create dynamic system prompt based on query analysis"""

        base_prompt = """You are Changi, an expert and friendly AI assistant for Changi Airport and Jewel Changi Airport in Singapore. You're known for being incredibly helpful, knowledgeable, and having a warm personality with just the right touch of playfulness while maintaining professionalism.

CORE RESPONSIBILITIES:
- Provide accurate, helpful information about Changi Airport and Jewel facilities
- Use ONLY the provided context to answer questions
- Be specific with details like operating hours, locations, terminal numbers, and contact information
- If information isn't available in the context, clearly state this limitation
- Always cite your sources when providing specific details

PERSONALITY TRAITS:
- Warm and welcoming, like a knowledgeable local friend
- Professional yet approachable
- Occasionally playful but never inappropriate
- Proactive in offering helpful suggestions
- Empathetic to traveler needs and stress

RESPONSE STRUCTURE:
1. Direct answer to the user's question
2. Relevant additional details that might be helpful
3. Source citations for specific information
4. A thoughtful follow-up suggestion or question

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

Remember: You're not just providing information, you're helping create a positive airport experience. Be genuinely helpful and make travelers feel welcomed and well-informed about their time at Changi Airport."""

        return base_prompt

    async def chat(self, user_message: str) -> Dict[str, Any]:
        """Main chat function with advanced processing"""

        try:
            # Analyze the query
            analysis = self.query_analyzer.analyze_query(user_message)

            # Retrieve relevant information
            try:
                retrieved_docs = await self.retriever.retrieve(user_message, top_k=8)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                retrieved_docs = []

            # Build context from retrieved documents
            context = self._build_context(retrieved_docs)

            # Get conversation history
            history = self.conversation_history[-6:]

            # Create system prompt
            system_prompt = self._create_system_prompt(analysis)

            # Build messages for the LLM
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history
            for msg in history:
                messages.append(msg)

            # Add current query with context
            if context.strip():
                user_prompt = f"""Context Information:
{context}

User Question: {user_message}

Please provide a helpful, accurate response based on the context provided. Include specific details and cite sources when mentioning particular facilities, services, or information."""
            else:
                user_prompt = f"""User Question: {user_message}

I don't have specific context information available right now, but please provide a helpful general response about Changi Airport based on your knowledge. Let the user know that for the most current and detailed information, they should check the official Changi Airport website or visit an information counter."""

            messages.append({"role": "user", "content": user_prompt})

            # Generate response
            try:
                if not self.client:
                    raise Exception("OpenAI client not initialized")
                    
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1500,
                    top_p=1,
                    stream=False
                )

                ai_response = response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI/Groq API error: {e}")
                ai_response = f"I'm having trouble connecting to my knowledge base right now. For questions about Changi Airport, I'd recommend checking the official Changi Airport website at changiairport.com or visiting one of the information counters located throughout the terminals."

            # Add a personalized suggestion
            suggestion = self._generate_suggestion(analysis, retrieved_docs)
            if suggestion:
                ai_response += f"\n\n💡 {suggestion}"

            # Update conversation memory
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Keep only recent history
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]

            return {
                "response": ai_response,
                "sources": [doc.source for doc in retrieved_docs[:3]] if retrieved_docs else [],
                "query_analysis": analysis.__dict__,
                "retrieved_count": len(retrieved_docs)
            }

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                "response": "I apologize, but I'm experiencing some technical difficulties right now. Please try asking your question again, or feel free to visit the information counter for immediate assistance.",
                "sources": [],
                "query_analysis": {},
                "retrieved_count": 0
            }

    def _build_context(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []

        for i, doc in enumerate(retrieved_docs[:6], 1):
            context_part = f"""Source {i} ({doc.category} - {doc.source}):
{doc.content[:800]}...
---"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _generate_suggestion(self, analysis: QueryAnalysis, retrieved_docs: List[RetrievalResult]) -> str:
        """Generate a personalized follow-up suggestion"""

        # Get dominant category from retrieved docs
        category_counts = defaultdict(int)
        for doc in retrieved_docs:
            category_counts[doc.category] += 1

        if not category_counts:
            return random.choice([
                "Is there anything specific about Changi Airport you'd like to explore?",
                "Would you like recommendations for your time at the airport?",
                "Feel free to ask about dining, shopping, or services at Changi!"
            ])

        dominant_category = max(category_counts, key=category_counts.get)

        # Map categories to suggestion categories
        suggestion_category = dominant_category
        if dominant_category == 'shops-retail':
            suggestion_category = 'shopping'
        elif dominant_category in ['attractions']:
            if dominant_category not in self.suggestions:
                suggestion_category = 'attractions'

        # Get suggestions for the category
        category_suggestions = self.suggestions.get(suggestion_category, self.suggestions['dining'])

        return random.choice(category_suggestions)

# ===============================================
# FASTAPI APPLICATION
# ===============================================

# Request/Response models
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

# Global chatbot instance
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global chatbot
    
    # Startup
    try:
        # Set up model caching directories
        cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME', '/tmp/sentence_transformers')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variables for model caching
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_HUB_CACHE'] = cache_dir
        
        # Get API keys from environment variables
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        INDEX_NAME = os.getenv("INDEX_NAME", "airportchatbot")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
        if not PINECONE_API_KEY or not GROQ_API_KEY:
            logger.warning("Missing API keys - running in demo mode")
            yield
            return
        
        # Initialize chatbot
        logger.info("Initializing Changi Airport Chatbot...")
        chatbot = ChangiAirportChatbot(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=INDEX_NAME,
            groq_api_key=GROQ_API_KEY
        )
        
        logger.info("✅ Chatbot initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        chatbot = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Changi Airport Chatbot API",
    description="RAG-based chatbot for Changi Airport and Jewel information",
    version="1.0.0",
    lifespan=lifespan
)

# Enhanced CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "https://airport-chatbot-front-end.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
    ],
    expose_headers=["*"]
)

# Manual CORS preflight handler
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests"""
    return {
        "message": "OK",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "headers": ["*"]
    }

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if chatbot else "starting",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    if not chatbot:
        return ChatResponse(
            response="I'm still starting up! Please wait a moment and try again. If this persists, my services might be temporarily unavailable.",
            sources=[],
            session_id=request.session_id or "default",
            query_analysis={},
            retrieved_count=0,
            timestamp=datetime.now().isoformat()
        )
    
    try:
        logger.info(f"Processing chat request: {request.message[:50]}...")
        
        result = await chatbot.chat(request.message)
        
        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            session_id=request.session_id or "default",
            query_analysis=result.get("query_analysis", {}),
            retrieved_count=result.get("retrieved_count", 0),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return ChatResponse(
            response="I'm experiencing some technical difficulties. Please try again in a moment.",
            sources=[],
            session_id=request.session_id or "default",
            query_analysis={},
            retrieved_count=0,
            timestamp=datetime.now().isoformat()
        )

@app.get("/categories")
async def get_categories():
    """Get available content categories"""
    return {
        "categories": [
            "dining",
            "shops-retail", 
            "services",
            "transportation",
            "attractions"
        ],
        "locations": [
            "terminal 1",
            "terminal 2", 
            "terminal 3",
            "terminal 4",
            "jewel",
            "transit area"
        ]
    }

@app.post("/search")
async def direct_search(request: ChatRequest):
    """Direct search functionality"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Use the retriever directly for search
        retrieved_docs = await chatbot.retriever.retrieve(request.message, top_k=10)
        
        return {
            "query": request.message,
            "results": [
                {
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "category": doc.category,
                    "source": doc.source,
                    "score": doc.score
                }
                for doc in retrieved_docs
            ],
            "total_results": len(retrieved_docs)
        }
        
    except Exception as e:
        logger.error(f"Error in direct search: {e}")
        raise HTTPException(status_code=500, detail="Search functionality unavailable")

@app.get("/cors-test")
async def cors_test():
    """Test CORS configuration"""
    return {
        "message": "CORS is working!",
        "timestamp": datetime.now().isoformat(),
        "headers_received": "OK"
    }

# For deployment
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
