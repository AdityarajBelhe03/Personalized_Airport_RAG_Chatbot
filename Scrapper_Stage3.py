# Crawler and main execution
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

GROQ_API_KEY = ' '

class ChangiCrawler:
    def __init__(self):
        self.config = BrowserConfig(headless=True, verbose=False)

    async def crawl_page(self, url: str):
        async with AsyncWebCrawler(config=self.config) as crawler:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=50
                )
            )
            return result.markdown

async def run_enhanced_scraping():
    """Main scraping function"""

    crawler = ChangiCrawler()
    processor = EnhancedGroqProcessor(GROQ_API_KEY)

    # Load URL config
    with open("config/urls.json", "r") as f:
        url_config = json.load(f)

    all_collections = {}

    # Process each collection
    for collection_type, config in url_config.items():
        print(f"Scraping collection: {collection_type}")
        extracted_items = []

        base_url = config["base_url"]
        pages = config["pages"]

        for page in pages:
            url = base_url + page
            print(f"  → Processing: {url}")

            try:
                # Crawl the page
                content = await crawler.crawl_page(url)

                # Extract data using enhanced processor
                items = await processor.process_with_enhanced_extraction(content, collection_type)

                # Add metadata
                for item in items:
                    item['source_url'] = url
                    item['collection'] = collection_type

                extracted_items.extend(items)
                print(f"     Extracted {len(items)} items")

                # Small delay to be respectful
                await asyncio.sleep(1)

            except Exception as e:
                print(f"    Failed to process {url}: {e}")

        # Save collection data
        collection_file = f"data/collections/{collection_type}.json"
        with open(collection_file, "w", encoding="utf-8") as f:
            json.dump(extracted_items, f, indent=2, ensure_ascii=False)

        all_collections[collection_type] = extracted_items
        print(f" Saved {len(extracted_items)} items to {collection_file}")

    print("\n Scraping completed!")

    # Summary
    total_items = sum(len(items) for items in all_collections.values())
    print(f"Total items extracted: {total_items}")

    for collection, items in all_collections.items():
        print(f"  - {collection}: {len(items)} items")

    return all_collections

# Run the scraping
result = await run_enhanced_scraping()
