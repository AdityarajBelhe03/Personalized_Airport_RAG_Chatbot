# Changi Airport Chatbot Backend

A sophisticated RAG (Retrieval-Augmented Generation) chatbot backend system for Singapore's Changi Airport and Jewel Changi Airport, powered by modern LLM technologies and vector databases.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

This backend system provides intelligent, context-aware responses about Changi Airport facilities, services, dining, shopping, and attractions. It combines web scraping, advanced natural language processing, vector embeddings, and conversational AI to deliver accurate information to travelers.

## Features

### Core Capabilities
- **Multi-Domain RAG System**: Intelligent retrieval across shopping, dining, services, transportation, and attractions
- **Advanced Query Analysis**: Sophisticated understanding of user intent and context
- **Conversation Memory**: Maintains context across multiple exchanges
- **Real-time Information Retrieval**: Fast, accurate responses with source attribution
- **Multi-Namespace Vector Search**: Optimized search across categorized content

### Technical Features
- **Async FastAPI Backend**: High-performance, scalable API server
- **Vector Database Integration**: Pinecone serverless for efficient similarity search
- **Smart Content Processing**: Category-specific data extraction and enhancement
- **Flexible Deployment**: Cloud-ready with environment-based configuration
- **Comprehensive Logging**: Structured logging for monitoring and debugging

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Scraping  │───▶│   Preprocessing  │───▶│   Embedding     │
│   (Crawl4AI)    │    │   & Cleaning     │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │◀───│   RAG Chatbot    │◀───│   Pinecone      │
│   REST API      │    │   Engine         │    │   Vector DB     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Overview

1. **Data Pipeline**
   - Web scraping with Crawl4AI
   - Content preprocessing and cleaning
   - Embedding generation with Sentence Transformers
   - Vector database storage in Pinecone

2. **RAG Engine**
   - Advanced query analysis and intent detection
   - Multi-domain retrieval strategies
   - Context-aware response generation
   - Conversation memory management

3. **API Layer**
   - RESTful endpoints with FastAPI
   - Request validation with Pydantic
   - Error handling and logging
   - CORS support for frontend integration

## Installation

### Prerequisites

- Python 3.11+
- pip or poetry
- Pinecone account and API key
- Groq API key

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/changi-chatbot-backend.git
cd changi-chatbot-backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the data pipeline** (Optional - for fresh data)
```bash
# Stage 1: Web scraping
python Scrapper_Stage1.py
python Scrapper_Stage2.py
python Scrapper_Stage3.py

# Stage 2: Preprocessing and embedding
python Preprocessing_Embedding.py
```

5. **Start the server**
```bash
python main.py
# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required API Keys
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration
INDEX_NAME=airportchatbot

# Server Configuration
PORT=8000
HOST=0.0.0.0

# Optional: Development settings
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Configuration Options

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PINECONE_API_KEY` | Pinecone vector database API key | - | Yes |
| `GROQ_API_KEY` | Groq LLM API key | - | Yes |
| `INDEX_NAME` | Pinecone index name | `airportchatbot` | No |
| `PORT` | Server port | `8000` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |

## Usage

### Starting the Server

```bash
# Development
python main.py

# Production with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Basic API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Chat with the bot
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What are the best halal dining options at Terminal 3?",
        "session_id": "user-123"
    }
)
print(response.json())
```

## API Documentation

### Endpoints

#### Health Check
```
GET /health
```
Returns server health status and version information.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0"
}
```

#### Chat
```
POST /chat
```
Main chatbot endpoint for conversational interactions.

**Request Body:**
```json
{
    "message": "string",
    "session_id": "string (optional)"
}
```

**Response:**
```json
{
    "response": "Detailed response text",
    "sources": ["source1", "source2"],
    "session_id": "user_session_id",
    "query_analysis": {
        "query_type": "complex",
        "categories": ["dining"],
        "location_filters": ["terminal 3"]
    },
    "retrieved_count": 5,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Direct Search
```
POST /search
```
Direct search functionality without conversational context.

**Request Body:**
```json
{
    "message": "search query"
}
```

**Response:**
```json
{
    "query": "search query",
    "results": [
        {
            "content": "relevant content",
            "category": "dining",
            "source": "changi_airport",
            "score": 0.85
        }
    ],
    "total_results": 10
}
```

#### Categories
```
GET /categories
```
Returns available content categories and locations.

**Response:**
```json
{
    "categories": ["dining", "shops-retail", "services", "transportation", "attractions"],
    "locations": ["terminal 1", "terminal 2", "terminal 3", "terminal 4", "jewel"]
}
```

### Error Handling

The API returns structured error responses:

```json
{
    "detail": "Error description",
    "status_code": 500,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

Common HTTP status codes:
- `200`: Success
- `422`: Validation Error
- `500`: Internal Server Error
- `503`: Service Unavailable

## Development

### Project Structure

```
changi-chatbot-backend/
├── main.py                 # FastAPI application and chatbot logic
├── Scrapper_Stage1.py      # Web scraping configuration
├── Scrapper_Stage2.py      # Content extraction logic
├── Scrapper_Stage3.py      # Scraping execution
├── Preprocessing_Embedding.py  # Data processing and embedding
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── config/                # Configuration files
│   └── urls.json         # Scraping target URLs
├── data/                 # Data storage
│   ├── collections/      # Raw scraped data
│   └── processed/        # Processed data and reports
└── logs/                 # Application logs
```

### Key Classes

#### `ChangiAirportChatbot`
Main chatbot class with RAG capabilities.

```python
chatbot = ChangiAirportChatbot(
    pinecone_api_key="your_key",
    index_name="airportchatbot",
    groq_api_key="your_groq_key"
)

result = await chatbot.chat("What dining options are available?")
```

#### `MultiDomainRetriever`
Advanced retrieval system for multi-namespace queries.

```python
retriever = MultiDomainRetriever(index, embedding_model, query_analyzer)
docs = await retriever.retrieve("query", top_k=10)
```

#### `AdvancedQueryAnalyzer`
Sophisticated query understanding and categorization.

```python
analyzer = AdvancedQueryAnalyzer()
analysis = analyzer.analyze_query("Find halal restaurants in Terminal 3")
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Development Setup

1. **Install development dependencies**
```bash
pip install -r requirements-dev.txt
```

2. **Set up pre-commit hooks**
```bash
pre-commit install
```

3. **Run linting**
```bash
black .
flake8 .
mypy .
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t changi-chatbot-backend .
docker run -p 8000:8000 --env-file .env changi-chatbot-backend
```

### Cloud Deployment (Render)

1. **Connect your GitHub repository to Render**
2. **Set environment variables in Render dashboard**
3. **Deploy with the following settings:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3.11

### Production Considerations

- **Scaling:** Use multiple worker processes with Gunicorn
- **Monitoring:** Implement health checks and logging
- **Security:** Use HTTPS and secure API key storage
- **Rate Limiting:** Implement rate limiting for production use
- **Caching:** Add Redis caching for frequently accessed data

## Performance Optimization

### Embedding Generation
- Batch processing for multiple queries
- GPU acceleration when available
- Caching for repeated queries

### Vector Search
- Namespace-based query optimization
- Smart filtering to reduce search space
- Result caching for common queries

### API Performance
- Async/await for concurrent request handling
- Connection pooling for database operations
- Response compression for large payloads

## Monitoring and Logging

### Logging Configuration

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chatbot.log'),
        logging.StreamHandler()
    ]
)
```

### Key Metrics to Monitor

- Response time per endpoint
- Query processing time
- Vector database query performance
- Error rates and types
- Memory usage and CPU utilization

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make your changes and add tests**
4. **Run the test suite:** `pytest`
5. **Run linting:** `black . && flake8 .`
6. **Commit your changes:** `git commit -m 'Add amazing feature'`
7. **Push to the branch:** `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for public methods
- Write comprehensive tests for new features

### Issues

Please use GitHub Issues to report bugs or request features. Include:
- Clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)


## Acknowledgments

- Singapore's Changi Airport for providing comprehensive public information
- Jewel Changi Airport for attraction and facility details
- Open source community for the excellent tools and libraries used in this project