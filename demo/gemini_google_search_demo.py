import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.gemini_google_search import GeminiPaperSummaryClient


def generate(model_name: str) -> str:
    client = GeminiPaperSummaryClient()
    return client.generate_summary(model_name=model_name)


if __name__ == "__main__":
    response = generate(model_name=sys.argv[1])
    print(response)
