import os
from firecrawl import Firecrawl
from dotenv import load_dotenv
load_dotenv()

app = Firecrawl(api_key=os.environ.get("FIRECRAWL_API_KEY"))

# Scrape a website and capture the result
doc = app.scrape('https://mlcontests.com/state-of-machine-learning-competitions-2024', formats=["markdown"])

# Print the results
print("=== MARKDOWN CONTENT ===")
print(doc.markdown[:500] if doc.markdown else "No markdown content")  # First 500 chars

print("\n=== METADATA ===")
print(f"Title: {doc.metadata.title if doc.metadata else 'N/A'}")
print(f"Description: {doc.metadata.description if doc.metadata else 'N/A'}")
print(f"Source URL: {doc.metadata.source_url if doc.metadata else 'N/A'}")

# Optional: Save to a file
if doc.markdown:
    with open('scraped_content.md', 'w', encoding='utf-8') as f:
        f.write(doc.markdown)
    print("\n✓ Saved markdown content to scraped_content.md")
