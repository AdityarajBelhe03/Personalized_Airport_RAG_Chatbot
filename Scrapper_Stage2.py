# Enhanced Processor Code (your complete enhanced code)
import asyncio
import json
import re
from typing import Dict, List, Optional
import logging
from groq import Groq

class EnhancedGroqProcessor:
    """Enhanced processor with better prompts for richer data extraction"""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def get_enhanced_extraction_prompt(self, collection_type: str) -> str:
        """Enhanced prompts for richer data extraction"""

        if collection_type == "attractions_and_entertainment":
            return """
You are an expert content extractor for Singapore's Changi Airport and Jewel Changi Airport.

Extract detailed attraction information and return ONLY a JSON array:

[
  {
    "attraction_name": "Full official name",
    "attraction_type": "Interactive/Nature/Experience/Entertainment/Outdoor",
    "location": "Specific location (e.g., 'Level 5, Canopy Park' or 'Basement 4, Forest Valley')",
    "operating_hours": "Detailed hours (e.g., '10:00 AM - 10:00 PM daily') or 'Not specified'",
    "ticket_pricing": "Detailed pricing (e.g., 'Adults $5, Children $3, Free for under 3') or 'Free' or 'Not specified'",
    "accessibility": "Wheelchair access, age restrictions, special requirements or 'Not specified'",
    "description": "Comprehensive description including what visitors can expect, unique features, duration, and experience highlights (minimum 50 words)"
  }
]

IMPORTANT RULES:
- Extract 5-15 attractions maximum
- Focus on unique, specific details about each attraction
- Include practical visitor information when available
- For Jewel attractions, mention connection to airport terminals if stated
- Descriptions must be detailed and informative, not generic
- Return only valid JSON, no explanatory text

Extract from this content focusing on visitor experience and practical details:
"""

        elif collection_type == "shops_and_retail":
            return """
You are an expert content extractor for Singapore's Changi Airport and Jewel Changi Airport shopping.

Extract detailed shop information and return ONLY a JSON array:

[
  {
    "shop_name": "Official store name",
    "category": "Fashion/Luxury/Electronics/Beauty/Food/Duty-Free/Books/Gifts/Other",
    "brand_type": "International/Local/Exclusive/Designer/Budget",
    "location": "Specific location (e.g., 'Terminal 3 Departure Hall Level 2' or 'Jewel Level 1')",
    "terminal": "T1/T2/T3/T4/Jewel/Multiple",
    "operating_hours": "Store hours or 'Not specified'",
    "special_features": "Tax-free shopping, exclusive items, airport-only products, or 'None'",
    "description": "Detailed description including brands carried, product range, unique offerings, and shopping experience (minimum 40 words)"
  }
]

IMPORTANT RULES:
- Extract 10-20 shops maximum
- Focus on distinctive shopping options and brands
- Include duty-free and tax advantages when mentioned
- Return only valid JSON, no explanatory text

Extract comprehensive shopping information from this content:
"""

        elif collection_type == "dining_options":
            return """
You are an expert content extractor for Singapore's Changi Airport and Jewel Changi Airport dining.

Extract detailed restaurant information and return ONLY a JSON array:

[
  {
    "restaurant_name": "Official restaurant name",
    "cuisine_type": "Asian/Western/Local/International/Fast Food/Fusion/Other",
    "dining_category": "Fine Dining/Casual Dining/Quick Service/Cafe/Bar/Food Court",
    "location": "Specific location with terminal and level",
    "terminal": "T1/T2/T3/T4/Jewel/Multiple",
    "operating_hours": "Restaurant hours or 'Not specified'",
    "price_range": "Budget ($)/Moderate ($$)/Expensive ($$$) or 'Not specified'",
    "special_dietary": "Halal/Vegetarian/Vegan/Gluten-free options or 'Not specified'",
    "signature_dishes": "Popular or recommended dishes, or 'Not specified'",
    "description": "Detailed description including ambiance, cuisine highlights, dining experience, and what makes it special (minimum 40 words)"
  }
]

IMPORTANT RULES:
- Extract 10-25 restaurants maximum
- Include food courts, cafes, and fine dining equally
- Focus on what makes each dining option unique
- Return only valid JSON, no explanatory text

Extract comprehensive dining information from this content:
"""

        elif collection_type == "baggage_and_services":
            return """
You are an expert content extractor for travel services at Singapore's Changi Airport.

Extract detailed baggage-and-service information and return ONLY a JSON array:

[
  {
    "service_name": "Official service name",
    "service_category": "Baggage/Storage/Check-in/Lost & Found/Security/Assistance/Other",
    "location": "Exact location with terminal and level",
    "terminal": "T1/T2/T3/T4/All/Not specified",
    "operating_hours": "Service hours or '24 hrs' or 'Not specified'",
    "pricing": "Detailed pricing or 'Free' or 'Not specified'",
    "procedures": "Key steps travellers must follow",
    "restrictions": "Weight/size limits, prohibited items, etc. or 'Not specified'",
    "contact_info": "Phone/email/desk number or 'Not specified'",
    "description": "Comprehensive description explaining what the service offers and when passengers should use it (minimum 40 words)"
  }
]

IMPORTANT RULES:
- Extract ALL relevant services on the page
- Include early check-in, baggage wrapping, porter services, etc.
- Return only valid JSON, no explanatory text

Extract comprehensive baggage & service info:
"""

        elif collection_type == "transportation_and_access":
            return """
You are an expert content extractor for ground transportation at Singapore's Changi Airport.

Extract detailed transportation information and return ONLY a JSON array:

[
  {
    "transport_type": "MRT/Bus/Taxi/Car/Coach/Shuttle/Other",
    "transport_category": "Public/Private/Shuttle/Transfer",
    "routes": "Route numbers or names, or 'Not specified'",
    "pickup_dropoff_points": "Exact locations (e.g., 'Basement 2 of T3')",
    "terminal": "T1/T2/T3/T4/Jewel/All/Not specified",
    "schedules": "Operating times or frequency details",
    "costs": "Fare ranges, surcharges, tolls, or 'Free'",
    "duration": "Typical travel times or 'Not specified'",
    "accessibility": "Wheelchair access, child-seat info, etc. or 'Not specified'",
    "description": "Detailed overview including when to choose this option, ticket-purchase tips, and any restrictions (minimum 40 words)"
  }
]

IMPORTANT RULES:
- Extract ALL transport modes mentioned
- Differentiate between landside and airside connections when possible
- Return only valid JSON, no explanatory text

Extract comprehensive transport & access info:
"""

        else:
            return f"""
Extract detailed {collection_type.replace('_', ' ')} information with rich descriptions and practical details.
Focus on specific, actionable information that would help airport visitors.
Include operating hours, locations, pricing, and unique features when available.
Descriptions should be detailed and informative (minimum 30 words each).
Return only valid JSON array format.
"""

    async def process_with_enhanced_extraction(self, content: str, collection_type: str) -> List[Dict]:
        """Process content with enhanced prompts for richer data"""

        # Pre-process content to focus on relevant sections
        relevant_content = self.extract_relevant_sections(content, collection_type)

        # Use enhanced prompt
        prompt = self.get_enhanced_extraction_prompt(collection_type)

        try:
            response = await asyncio.to_thread(
                self._make_enhanced_request,
                prompt,
                relevant_content
            )

            if response and response.choices:
                extracted_text = response.choices[0].message.content.strip()
                parsed_data = self.parse_json_flexible(extracted_text)

                # Post-process to ensure quality
                quality_data = self.ensure_data_quality(parsed_data, collection_type)
                return quality_data

        except Exception as e:
            logging.error(f"Enhanced extraction failed: {e}")
            return []

    def extract_relevant_sections(self, content: str, collection_type: str) -> str:
        """Extract most relevant content sections for better processing"""

        # Define keywords for each collection type
        keywords = {
            "attractions_and_entertainment": [
                "attraction", "experience", "park", "garden", "vortex", "forest",
                "canopy", "slides", "maze", "studio", "interactive", "play"
            ],
            "shops_and_retail": [
                "shop", "store", "boutique", "retail", "brand", "fashion",
                "luxury", "duty", "tax", "electronics", "beauty"
            ],
            "dining_options": [
                "restaurant", "cafe", "dining", "food", "cuisine", "bar",
                "menu", "halal", "breakfast", "lunch", "dinner"
            ],
            "baggage_and_services": [
                "baggage", "luggage", "check-in", "wrapping", "storage", "locker",
                "porter", "lost and found", "tracking", "allowance", "weight limit",
                "oversized", "customs", "security", "immigration", "assistance"
            ],
            "transportation_and_access": [
                "mrt", "metro", "train", "bus", "taxi", "grab", "parking",
                "car rental", "shuttle", "coach", "skytrain", "people mover",
                "route", "pickup", "drop-off", "fare", "ez-link", "duration"
            ]
        }

        relevant_keywords = keywords.get(collection_type, [])

        # Split content into paragraphs and score relevance
        paragraphs = content.split('\n')
        scored_paragraphs = []

        for para in paragraphs:
            if len(para.strip()) < 20:
                continue

            score = sum(1 for keyword in relevant_keywords
                       if keyword.lower() in para.lower())

            if score > 0:
                scored_paragraphs.append((score, para))

        # Sort by relevance and take top paragraphs
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        top_content = '\n'.join([para for _, para in scored_paragraphs[:10]])

        # If no relevant content found, return original (truncated)
        return top_content if top_content.strip() else content[:4000]

    def ensure_data_quality(self, data: List[Dict], collection_type: str) -> List[Dict]:
        """Ensure extracted data meets quality standards"""

        quality_data = []

        for item in data:
            # Check for minimum description length
            description = item.get('description', '')
            if len(description.split()) < 8:  # Less than 8 words
                item['description'] = f"Information about {item.get(list(item.keys())[0], 'this item')} available at the location."

            # Validate required fields exist
            if collection_type == "attractions_and_entertainment":
                if not item.get('attraction_name'):
                    continue

            quality_data.append(item)

        return quality_data

    def _make_enhanced_request(self, prompt: str, content: str):
        """Make enhanced Groq request with higher token limit"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a meticulous data extraction expert specializing in travel and tourism information. Provide comprehensive, accurate details."},
                {"role": "user", "content": f"{prompt}\n\nContent:\n{content}"}
            ],
            temperature=0.1,
            max_tokens=2000,
            top_p=0.9
        )

    def parse_json_flexible(self, text: str) -> List[Dict]:
        """Enhanced JSON parsing with error recovery"""
        text = text.strip()

        # Remove code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'\s*```', '', text)

        # Try direct parsing
        try:
            if text.startswith('['):
                return json.loads(text)
            elif text.startswith('{'):
                return [json.loads(text)]
        except json.JSONDecodeError:
            pass

        # Extract JSON arrays or objects
        json_pattern = r'\[[\s\S]*?\]|\{[\s\S]*?\}'
        matches = re.findall(json_pattern, text)

        for match in matches:
            try:
                data = json.loads(match)
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                continue

        return []

print("✅ Enhanced processor loaded!")