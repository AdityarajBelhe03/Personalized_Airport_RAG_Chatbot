#preprocessing and ebedding stage

import json
import os
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Core libraries
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChangiDataPreprocessor:
    """
    Preprocesses Changi Airport JSON data for optimal embedding generation and Pinecone upsert
    """

    def __init__(self):
        self.collection_mapping = {
            'shops_and_retail': 'shops-retail',
            'attractions_and_entertainment': 'attractions',
            'dining_options': 'dining',
            'baggage_and_services': 'services',
            'transportation_and_access': 'transportation'
        }

        # Content processing strategies per collection
        self.text_strategies = {
            'shops_and_retail': self._process_shop_text,
            'attractions_and_entertainment': self._process_attraction_text,
            'dining_options': self._process_dining_text,
            'baggage_and_services': self._process_service_text,
            'transportation_and_access': self._process_transport_text
        }

    def load_json_files(self, data_dir: str = "/content/data/collections") -> Dict[str, List[Dict]]:
        """Load all 5 JSON files from structured data directory"""
        collections_data = {}
        data_path = Path(data_dir)

        for collection_type in self.collection_mapping.keys():
            file_path = data_path / f"{collection_type}.json"

            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    collections_data[collection_type] = data
                    logger.info(f" Loaded {collection_type}: {len(data)} items")
            else:
                logger.warning(f" File not found: {file_path}")
                collections_data[collection_type] = []

        return collections_data

    def deduplicate_data(self, collections_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Remove duplicates within and across collections"""
        seen_hashes = set()
        deduplicated = {}

        for collection_type, items in collections_data.items():
            unique_items = []

            for item in items:
                # Create content hash for deduplication
                content = self._extract_core_content(item, collection_type)
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    item['content_hash'] = content_hash
                    unique_items.append(item)

            deduplicated[collection_type] = unique_items
            logger.info(f"🔍 {collection_type}: {len(items)} → {len(unique_items)} (removed {len(items) - len(unique_items)} duplicates)")

        return deduplicated

    def chunk_and_process_data(self, collections_data: Dict[str, List[Dict]]) -> List[Dict]:
        """Process data into optimized chunks for embedding"""
        processed_chunks = []

        for collection_type, items in collections_data.items():
            logger.info(f" Processing {collection_type}...")

            strategy = self.text_strategies.get(collection_type, self._process_generic_text)

            for idx, item in enumerate(items):
                # Extract and process embedding text
                embedding_text = strategy(item)

                # Create chunk with metadata
                chunk = {
                    'id': f"{self.collection_mapping[collection_type]}_{item.get('content_hash', idx)}",
                    'text': embedding_text,  # Critical for retrieval!
                    'namespace': self.collection_mapping[collection_type],
                    'metadata': {
                        'collection_type': collection_type,
                        'namespace': self.collection_mapping[collection_type],
                        'source_site': self._extract_source_site(item.get('source_url', '')),
                        'terminal': self._extract_terminal(item),
                        'original_id': item.get('id', ''),
                        'source_url': item.get('source_url', ''),
                        'scraped_at': item.get('scraped_at', ''),
                        'text': embedding_text,  # Store in metadata too - IMPORTANT!
                        'content_hash': item.get('content_hash', ''),
                        # Store key original fields for retrieval
                        **self._extract_key_metadata(item, collection_type)
                    }
                }

                processed_chunks.append(chunk)

        logger.info(f" Total processed chunks: {len(processed_chunks)}")
        return processed_chunks

    def _extract_core_content(self, item: Dict, collection_type: str) -> str:
        """Extract core content for deduplication"""
        strategy = self.text_strategies.get(collection_type, self._process_generic_text)
        return strategy(item)

    def _process_shop_text(self, item: Dict) -> str:
        """Enhanced semantic processing for shops and retail"""
        semantic_parts = []

        # Primary identification
        shop_name = item.get('shop_name', '')
        if shop_name:
            semantic_parts.append(f"Store: {shop_name}")

        # Category classification
        category = item.get('category', '')
        brand_type = item.get('brand_type', '')
        if category:
            category_text = f"Category: {category}"
            if brand_type:
                category_text += f" ({brand_type})"
            semantic_parts.append(category_text)

        # Location specifics
        location = item.get('location', '')
        terminal = item.get('terminal', '')
        if location:
            loc_text = f"Location: {location}"
            if terminal and terminal not in location:
                loc_text += f" in {terminal}"
            semantic_parts.append(loc_text)

        # Floor information
        floor_level = item.get('floor_level', '')
        if floor_level and floor_level != "Not specified":
            semantic_parts.append(f"Floor: {floor_level}")

        # Operational details
        hours = item.get('operating_hours', '')
        if hours and hours != "Not specified":
            semantic_parts.append(f"Hours: {hours}")

        # Contact information
        contact = item.get('contact_info', '')
        if contact and contact != "Not specified":
            semantic_parts.append(f"Contact: {contact}")

        # Special features
        features = item.get('special_features', '')
        if features and features.lower() != 'none':
            semantic_parts.append(f"Features: {features}")

        # Description
        description = item.get('description', '')
        if description:
            semantic_parts.append(f"About: {description}")

        return ". ".join(semantic_parts)

    def _process_attraction_text(self, item: Dict) -> str:
        """Enhanced semantic processing for attractions and entertainment"""
        semantic_parts = []

        # Attraction identity
        attraction_name = item.get('attraction_name', '')
        attraction_type = item.get('attraction_type', '')
        if attraction_name:
            attraction_text = f"Attraction: {attraction_name}"
            if attraction_type:
                attraction_text += f" ({attraction_type})"
            semantic_parts.append(attraction_text)

        # Location details
        location = item.get('location', '')
        if location:
            semantic_parts.append(f"Location: {location}")

        # Operational information
        hours = item.get('operating_hours', '')
        if hours and hours != "Not specified":
            semantic_parts.append(f"Operating hours: {hours}")

        # Pricing
        ticket_pricing = item.get('ticket_pricing', '')
        if ticket_pricing and ticket_pricing != "Not specified":
            semantic_parts.append(f"Pricing: {ticket_pricing}")

        # Accessibility
        accessibility = item.get('accessibility', '')
        if accessibility and accessibility != "Not specified":
            semantic_parts.append(f"Accessibility: {accessibility}")

        # Description
        description = item.get('description', '')
        if description:
            semantic_parts.append(f"Experience: {description}")

        return ". ".join(semantic_parts)

    def _process_dining_text(self, item: Dict) -> str:
        """Enhanced semantic processing for dining options"""
        semantic_parts = []

        # Restaurant identity
        restaurant_name = item.get('restaurant_name', '')
        cuisine_type = item.get('cuisine_type', '')
        dining_category = item.get('dining_category', '')

        if restaurant_name:
            restaurant_text = f"Restaurant: {restaurant_name}"
            if cuisine_type:
                restaurant_text += f" ({cuisine_type})"
            semantic_parts.append(restaurant_text)

        # Dining type
        if dining_category:
            semantic_parts.append(f"Type: {dining_category}")

        # Location specifics
        location = item.get('location', '')
        terminal = item.get('terminal', '')
        if location:
            loc_text = f"Location: {location}"
            if terminal and terminal not in location:
                loc_text += f" in {terminal}"
            semantic_parts.append(loc_text)

        # Floor information
        floor_level = item.get('floor_level', '')
        if floor_level and floor_level != "Not specified":
            semantic_parts.append(f"Floor: {floor_level}")

        # Operational details
        hours = item.get('operating_hours', '')
        if hours and hours != "Not specified":
            semantic_parts.append(f"Hours: {hours}")

        # Pricing
        price_range = item.get('price_range', '')
        if price_range and price_range != "Not specified":
            semantic_parts.append(f"Price range: {price_range}")

        # Special dietary options
        special_dietary = item.get('special_dietary', '')
        if special_dietary and special_dietary != "Not specified":
            semantic_parts.append(f"Dietary options: {special_dietary}")

        # Description
        description = item.get('description', '')
        if description:
            semantic_parts.append(f"About: {description}")

        return ". ".join(semantic_parts)

    def _process_service_text(self, item: Dict) -> str:
        """Enhanced semantic processing for baggage and services"""
        semantic_parts = []

        # Service identity
        service_name = item.get('service_name', '')
        service_category = item.get('service_category', '')

        if service_name:
            service_text = f"Service: {service_name}"
            if service_category:
                service_text += f" ({service_category})"
            semantic_parts.append(service_text)

        # Location and availability
        location = item.get('location', '')
        terminal = item.get('terminal', '')
        if location:
            loc_info = f"Available at: {location}"
            if terminal and terminal not in location:
                loc_info += f" ({terminal})"
            semantic_parts.append(loc_info)

        # Operational information
        hours = item.get('operating_hours', '')
        if hours and hours != "Not specified":
            semantic_parts.append(f"Operating hours: {hours}")

        # Pricing
        pricing = item.get('pricing', '')
        if pricing and pricing != "Not specified":
            semantic_parts.append(f"Cost: {pricing}")

        # Procedures
        procedures = item.get('procedures', '')
        if procedures:
            semantic_parts.append(f"Process: {procedures}")

        # Restrictions
        restrictions = item.get('restrictions', '')
        if restrictions and restrictions != "Not specified":
            semantic_parts.append(f"Restrictions: {restrictions}")

        # Contact information
        contact = item.get('contact_info', '')
        if contact and contact != "Not specified":
            semantic_parts.append(f"Contact: {contact}")

        # Description
        description = item.get('description', '')
        if description:
            semantic_parts.append(f"Details: {description}")

        return ". ".join(semantic_parts)

    def _process_transport_text(self, item: Dict) -> str:
        """Enhanced semantic processing for transportation and access"""
        semantic_parts = []

        # Transport identity
        transport_type = item.get('transport_type', '')
        transport_category = item.get('transport_category', '')

        if transport_type:
            transport_text = f"Transport: {transport_type}"
            if transport_category:
                transport_text += f" ({transport_category})"
            semantic_parts.append(transport_text)

        # Route information
        routes = item.get('routes', '')
        if routes:
            semantic_parts.append(f"Routes: {routes}")

        # Location details
        locations = item.get('locations', '')
        if locations:
            semantic_parts.append(f"Stops: {locations}")

        # Schedule information
        schedules = item.get('schedules', '')
        if schedules and schedules != "Not specified":
            semantic_parts.append(f"Schedule: {schedules}")

        # Duration
        duration = item.get('duration', '')
        if duration and duration != "Not specified":
            semantic_parts.append(f"Duration: {duration}")

        # Cost information
        costs = item.get('costs', '')
        if costs and costs != "Not specified":
            semantic_parts.append(f"Cost: {costs}")

        # Accessibility
        accessibility = item.get('accessibility', '')
        if accessibility and accessibility != "Not specified":
            semantic_parts.append(f"Accessibility: {accessibility}")

        # Description
        description = item.get('description', '')
        if description:
            semantic_parts.append(f"Details: {description}")

        return ". ".join(semantic_parts)

    def _process_generic_text(self, item: Dict) -> str:
        """Generic text processing fallback"""
        return str(item.get('description', ''))

    def _clean_and_join(self, parts: List[str]) -> str:
        """Clean and join text parts"""
        # Filter out empty strings and "Not specified"
        clean_parts = [
            part.strip() for part in parts
            if part and part.strip() and part.strip().lower() != "not specified"
        ]

        # Join with spaces and clean
        text = ' '.join(clean_parts)

        # Clean whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ').replace('\r', '')

        # Truncate if too long (keep under 500 tokens for efficiency)
        if len(text) > 2000:
            text = text[:2000] + "..."

        return text.strip()

    def _extract_source_site(self, url: str) -> str:
        """Extract source site from URL"""
        if 'jewelchangiairport.com' in url:
            return 'jewel'
        elif 'changiairport.com' in url:
            return 'changi'
        return 'unknown'

    def _extract_terminal(self, item: Dict) -> str:
        """Extract terminal information"""
        # Check explicit terminal field first
        if item.get('terminal'):
            return item['terminal']

        # Extract from location
        location = item.get('location', '').lower()

        if 'terminal 1' in location or 't1' in location:
            return 'T1'
        elif 'terminal 2' in location or 't2' in location:
            return 'T2'
        elif 'terminal 3' in location or 't3' in location:
            return 'T3'
        elif 'terminal 4' in location or 't4' in location:
            return 'T4'
        elif 'jewel' in location or 'jewelchangiairport.com' in item.get('source_url', ''):
            return 'Jewel'

        return 'Changi Airport'

    def _extract_key_metadata(self, item: Dict, collection_type: str) -> Dict:
        """Extract key metadata fields for retrieval"""
        metadata = {}

        # Collection-specific key fields
        key_fields = {
            'shops_and_retail': ['shop_name', 'category'],
            'attractions_and_entertainment': ['attraction_name', 'attraction_type'],
            'dining_options': ['restaurant_name', 'cuisine_type', 'dining_category'],
            'baggage_and_services': ['service_name', 'service_category'],
            'transportation_and_access': ['transport_type', 'transport_category']
        }

        for field in key_fields.get(collection_type, []):
            if item.get(field):
                metadata[field] = item[field]

        # Common useful fields
        common_fields = ['location', 'operating_hours', 'contact_info']
        for field in common_fields:
            if item.get(field) and item[field] != "Not specified":
                metadata[field] = item[field]

        return metadata


class ChangiEmbeddingGenerator:
    """
    Generates embeddings using HuggingFace sentence-transformers
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f" Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(self, chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
        """Generate embeddings for all chunks"""
        logger.info(f" Generating embeddings for {len(chunks)} chunks...")

        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            embeddings.extend(batch_embeddings)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['values'] = embedding.tolist()

        logger.info(f" Generated {len(embeddings)} embeddings")
        return chunks


class PineconeUploader:
    """
    Handles Pinecone connection, namespace creation, and batch upserts
    """

    def __init__(self, api_key: str, index_name: str, cloud: str = "aws", region: str = "us-east-1"):
        self.api_key = api_key
        self.index_name = index_name
        self.cloud = cloud
        self.region = region

        # Initialize Pinecone with new SDK
        self.pc = Pinecone(api_key=api_key)

        # Connect to index
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if index_name not in existing_indexes:
                logger.error(f" Index '{index_name}' not found!")
                logger.info(f"Available indexes: {existing_indexes}")
                raise ValueError(f"Index '{index_name}' does not exist")

            self.index = self.pc.Index(index_name)
            logger.info(f" Connected to Pinecone index: {index_name}")

        except Exception as e:
            logger.error(f" Failed to connect to Pinecone: {e}")
            raise

    def create_index_if_not_exists(self, dimension: int = 384):
        """Create index if it doesn't exist (for setup)"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"🔧 Creating new index: {self.index_name}")

                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )

                # Wait for index to be ready
                import time
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)

                logger.info(f" Index '{self.index_name}' created successfully")

            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            logger.error(f" Failed to create index: {e}")
            raise

    def upsert_with_namespaces(self, chunks: List[Dict], batch_size: int = 100):
        """Upsert chunks to Pinecone using namespaces"""

        # Group chunks by namespace
        namespace_groups = {}
        for chunk in chunks:
            namespace = chunk['namespace']
            if namespace not in namespace_groups:
                namespace_groups[namespace] = []
            namespace_groups[namespace].append(chunk)

        logger.info(f" Upserting to {len(namespace_groups)} namespaces:")
        for ns, items in namespace_groups.items():
            logger.info(f"  {ns}: {len(items)} items")

        total_upserted = 0

        # Upsert each namespace separately
        for namespace, namespace_chunks in namespace_groups.items():
            logger.info(f" Upserting namespace: {namespace}")

            # Batch upsert within namespace
            for i in tqdm(range(0, len(namespace_chunks), batch_size),
                         desc=f"Upsetting {namespace}"):
                batch = namespace_chunks[i:i + batch_size]

                # Prepare vectors for Pinecone - FIXED: Include text in metadata
                vectors = []
                for chunk in batch:
                    # Ensure text is always in metadata for retrieval
                    metadata = chunk['metadata'].copy()
                    if 'text' not in metadata:
                        metadata['text'] = chunk['text']

                    vector = {
                        'id': chunk['id'],
                        'values': chunk['values'],
                        'metadata': metadata  # Now includes text!
                    }
                    vectors.append(vector)

                # Upsert with retry logic
                retries = 3
                for attempt in range(retries):
                    try:
                        self.index.upsert(vectors=vectors, namespace=namespace)
                        total_upserted += len(vectors)
                        break
                    except Exception as e:
                        if attempt == retries - 1:
                            logger.error(f"❌ Failed to upsert batch after {retries} attempts: {e}")
                            raise
                        else:
                            logger.warning(f"⚠ Upsert attempt {attempt + 1} failed, retrying...")
                            import time
                            time.sleep(2 ** attempt)  # Exponential backoff

        logger.info(f" Successfully upserted {total_upserted} vectors across {len(namespace_groups)} namespaces")

        # Verify upload
        self.verify_upload(namespace_groups)

    def verify_upload(self, namespace_groups: Dict[str, List[Dict]]):
        """Verify that data was uploaded correctly"""
        logger.info(" Verifying upload...")

        try:
            # Get index stats
            stats = self.index.describe_index_stats()

            for namespace, chunks in namespace_groups.items():
                if 'namespaces' in stats and namespace in stats['namespaces']:
                    uploaded_count = stats['namespaces'][namespace]['vector_count']
                    expected_count = len(chunks)

                    if uploaded_count == expected_count:
                        logger.info(f" {namespace}: {uploaded_count}/{expected_count} vectors")
                    else:
                        logger.warning(f" {namespace}: {uploaded_count}/{expected_count} vectors (mismatch)")
                else:
                    logger.warning(f" {namespace}: Namespace not found in stats (may need time to reflect)")

        except Exception as e:
            logger.warning(f" Could not verify upload: {e}")

    def test_query(self, namespace: str, text: str = "test query", top_k: int = 3):
        """Test query to verify namespace works"""
        try:
            # Simple test query (you'd normally use embeddings here)
            results = self.index.query(
                vector=[0.1] * 384,  # Dummy vector for testing
                namespace=namespace,
                top_k=top_k,
                include_metadata=True
            )

            logger.info(f"🔍 Test query for namespace '{namespace}': {len(results.matches)} results")
            return results

        except Exception as e:
            logger.error(f" Test query failed for namespace '{namespace}': {e}")
            return None


def main():
    """Main execution pipeline for Stage 2"""

    # Configuration - FIXED: Removed PINECONE_ENV, simplified
    DATA_DIR = "/content/data/collections"
    # Use direct values instead of environment variables for simplicity
    PINECONE_API_KEY = " "
    PINECONE_INDEX = "airportchatbot"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Validate API key
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not provided")

    logger.info(" Starting Changi Airport Data Pipeline - Stage 2")
    logger.info("=" * 60)

    try:
        # Step 1: Load and preprocess data
        logger.info(" Step 1: Loading JSON files...")
        preprocessor = ChangiDataPreprocessor()
        collections_data = preprocessor.load_json_files(DATA_DIR)

        if not any(collections_data.values()):
            raise ValueError("No data found in JSON files")

        # Step 2: Deduplicate
        logger.info(" Step 2: Deduplicating data...")
        clean_data = preprocessor.deduplicate_data(collections_data)

        # Step 3: Chunk and process
        logger.info("⚙ Step 3: Processing and chunking...")
        processed_chunks = preprocessor.chunk_and_process_data(clean_data)

        # Step 4: Generate embeddings
        logger.info(" Step 4: Generating embeddings...")
        embedding_generator = ChangiEmbeddingGenerator(EMBEDDING_MODEL)
        embedded_chunks = embedding_generator.generate_embeddings(processed_chunks)

        # Step 5: Upsert to Pinecone - FIXED: Removed PINECONE_ENV parameter
        logger.info(" Step 5: Upserting to Pinecone...")
        uploader = PineconeUploader(PINECONE_API_KEY, PINECONE_INDEX)
        uploader.upsert_with_namespaces(embedded_chunks)

        # Summary report
        namespace_summary = {}
        for chunk in embedded_chunks:
            ns = chunk['namespace']
            namespace_summary[ns] = namespace_summary.get(ns, 0) + 1

        logger.info("\n PIPELINE COMPLETE!")
        logger.info(" Summary:")
        logger.info(f"  Total Chunks: {len(embedded_chunks)}")
        logger.info(f"  Namespaces: {len(namespace_summary)}")
        for ns, count in namespace_summary.items():
            logger.info(f"    {ns}: {count} vectors")
        logger.info(f"  Embedding Model: {EMBEDDING_MODEL}")
        logger.info(f"  Pinecone Index: {PINECONE_INDEX}")

        # Save processing report
        report = {
            'processing_timestamp': datetime.now().isoformat(),
            'total_chunks': len(embedded_chunks),
            'namespace_summary': namespace_summary,
            'embedding_model': EMBEDDING_MODEL,
            'pinecone_index': PINECONE_INDEX,
            'data_source': DATA_DIR
        }

        report_path = Path("data/processed/stage2_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f" Processing report saved: {report_path}")

    except Exception as e:
        logger.error(f" Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
