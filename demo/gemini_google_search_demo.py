from tools.gemini_google_search import GeminiGoogleSearchClient
import sys


def generate(model_name: str) -> str:
    client = GeminiGoogleSearchClient()
    return client.generate_summary(model_name=model_name)


if __name__ == "__main__":
    response = generate(model_name=sys.argv[1])
    print(response)
