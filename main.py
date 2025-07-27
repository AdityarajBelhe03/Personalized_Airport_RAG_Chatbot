# Complete Changi Airport Chatbot - Single File for Deployment
import os
import json
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import re

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Core dependencies
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from openai import OpenAI
import numpy as np
from collections import defaultdict
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# CHATBOT CLASSES (Your existing RAG code)
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
        # FIXED: Match exact namespace names used during upsert
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
            categories=categories if categories else self.categories,  # Default to all if none specified
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
        # Simple keyword extraction (can be enhanced with NLP)
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

        # FIXED: Add normalize_embeddings=True to match upsert normalization
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

        # FIXED: Better distribution logic - only distribute if we have more categories than top_k
        if len(analysis.categories) > top_k:
            k_per_category = max(2, top_k // len(analysis.categories))
        else:
            k_per_category = top_k  # Request full amount from each category

        for category in analysis.categories:
            try:
                # Build filter for this category
                filter_dict = self._build_filter(analysis, category)

                # Query this namespace
                response = self.index.query(
                    vector=query_embedding,
                    top_k=k_per_category,
                    namespace=category,
                    filter=filter_dict if filter_dict else None,  # Only pass filter if not empty
                    include_metadata=True
                )

                # FIXED: Process results - use 'text' key instead of 'content'
                for match in response['matches']:
                    result = RetrievalResult(
                        content=match['metadata'].get('text', ''),  # FIXED: Changed from 'content' to 'text'
                        metadata=match['metadata'],
                        score=match['score'],
                        category=category,
                        source=match['metadata'].get('source', '')
                    )
                    all_results.append(result)

            except Exception as e:
                logger.error(f"Error retrieving from {category}: {e}")

        # Apply additional filtering and boost scores BEFORE sorting
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
                    filter=filter_dict if filter_dict else None,  # Only pass filter if not empty
                    include_metadata=True
                )

                # FIXED: Use 'text' key instead of 'content'
                for match in response['matches']:
                    result = RetrievalResult(
                        content=match['metadata'].get('text', ''),  # FIXED: Changed from 'content' to 'text'
                        metadata=match['metadata'],
                        score=match['score'],
                        category=category,
                        source=match['metadata'].get('source', '')
                    )
                    results.append(result)

            except Exception as e:
                logger.error(f"Error in complex retrieval for {category}: {e}")

        # Apply sophisticated filtering and boost scores
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
                filter=filter_dict if filter_dict else None,  # Only pass filter if not empty
                include_metadata=True
            )

            # FIXED: Use 'text' key instead of 'content'
            for match in response['matches']:
                result = RetrievalResult(
                    content=match['metadata'].get('text', ''),  # FIXED: Changed from 'content' to 'text'
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

        # FIXED: Location filtering - use $contains_any instead of $in for substring matching
        if analysis.location_filters:
            filter_dict['location'] = {'$contains_any': analysis.location_filters}

        # FIXED: Dietary requirements - use $contains_any instead of $regex
        if analysis.dietary_requirements and category == 'dining':
            filter_dict['dietary_info'] = {'$contains_any': analysis.dietary_requirements}

        return filter_dict

    def _apply_post_retrieval_filters(self, results: List[RetrievalResult], analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Apply additional filtering after retrieval"""
        filtered_results = []

        for result in results:
            # Price filtering
            if analysis.price_filters and 'max_price' in analysis.price_filters:
                price_text = result.content.lower()
                # Simple price extraction (can be enhanced)
                price_matches = re.findall(r'\$(\d+)', price_text)
                if price_matches:
                    prices = [int(p) for p in price_matches]
                    min_price = min(prices)
                    if min_price > analysis.price_filters['max_price']:
                        continue

            # FIXED: Keyword relevance boost - apply BEFORE final sorting
            content_lower = result.content.lower()
            keyword_matches = sum(1 for keyword in analysis.keywords if keyword in content_lower)
            if keyword_matches > 0:
                result.score *= (1 + keyword_matches * 0.1)  # Boost score

            filtered_results.append(result)

        return filtered_results

class ChangiAirportChatbot:
    """Main chatbot class with advanced conversational capabilities"""

    def __init__(self, pinecone_api_key: str, index_name: str, groq_api_key: str):
        # Initialize Pinecone with new v3+ API
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize components
        self.query_analyzer = AdvancedQueryAnalyzer()
        self.retriever = MultiDomainRetriever(self.index, self.embedding_model, self.query_analyzer)

        # Initialize conversation memory
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True
        )

        # FIXED: Initialize OpenAI client with Groq endpoint
        self.client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = "mixtral-8x7b-32768"  # Changed to a more reliable model

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
            history = self.memory.chat_memory.messages

            # Create system prompt
            system_prompt = self._create_system_prompt(analysis)

            # Build messages for the LLM
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history
            for msg in history[-6:]:  # Last 3 exchanges
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})

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
            self.memory.chat_memory.add_user_message(user_message)
            self.memory.chat_memory.add_ai_message(ai_response)

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
            # Check if we have suggestions for this category
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

# Initialize FastAPI app
app = FastAPI(
    title="Changi Airport Chatbot API",
    description="RAG-based chatbot for Changi Airport and Jewel information",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot
    
    try:
        # Get API keys from environment variables
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        INDEX_NAME = os.getenv("INDEX_NAME", "airportchatbot")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
        if not PINECONE_API_KEY or not GROQ_API_KEY:
            raise ValueError("Missing required API keys in environment variables")
        
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
        raise

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if chatbot else "unhealthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Process the chat request
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
        raise HTTPException(
            status_code=500, 
            detail="I'm experiencing some technical difficulties. Please try again."
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

# For deployment - get PORT from environment
PORT = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
