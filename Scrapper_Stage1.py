# Install dependencies
!pip install crawl4ai groq asyncio beautifulsoup4 python-dotenv sentence-transformers

# Create directory structure
import os
import json
from pathlib import Path

# Create necessary directories
Path("config").mkdir(exist_ok=True)
Path("data/collections").mkdir(parents=True, exist_ok=True)

# Set your GROQ API key directly in the code
GROQ_API_KEY = " "

# Create URL configuration
url_config = {
    "shops_and_retail": {
        "base_url": "https://www.jewelchangiairport.com",
        "pages": ["/shop", "/shop/directory", "/shop/fashion-beauty"]
    },
    "attractions_and_entertainment": {
        "base_url": "https://www.jewelchangiairport.com",
        "pages": ["/attractions", "/attractions/rain-vortex", "/attractions/forest-valley"]
    },
    "dining_options": {
        "base_url": "https://www.jewelchangiairport.com",
        "pages": ["/dine", "/dine/restaurants", "/dine/cafes-bars"]
    },
    "baggage_and_services": {
        "base_url": "https://www.changiairport.com",
        "pages": ["/at-changi/baggage-services", "/at-changi/baggage-tracking"]
    },
    "transportation_and_access": {
        "base_url": "https://www.changiairport.com",
        "pages": ["/to-from-airport", "/to-from-airport/public-transport"]
    }
}

# Save config
with open("config/urls.json", "w") as f:
    json.dump(url_config, f, indent=2)

print("✅ Setup complete!")