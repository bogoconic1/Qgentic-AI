from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def _safe_read(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""

def ask_domain_expert(question: str):
    slug = os.environ.get("TASK_SLUG", "make-data-count-finding-data-references")
    base_dir = os.path.join("task", slug, "threads")
    if os.path.isdir(base_dir):
        threads = [os.path.join(base_dir, p) for p in os.listdir(base_dir) if p.endswith('.md')]
        kb_threads = "\n------------------------\n".join([_safe_read(p) for p in threads])
    else:
        kb_threads = ""

    docs_dir = os.path.join("task", slug)
    overview = _safe_read(os.path.join(docs_dir, "overview.md"))
    data_description = _safe_read(os.path.join(docs_dir, "data_description.md"))

    SYS_PROMPT = f"""You are a domain expert answering questions about a specific competition.
Competition Overview (from docs):\n{overview}\n\nData Description (from docs):\n{data_description}\n\nAdditional Knowledge Base (community threads):\n{kb_threads}\n\nGuidelines:\n- Prefer the official docs content when available.\n- If the answer is not supported by the provided materials, respond with "I don't know".\n"""

    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT
        },
        {
            "role": "user",
            "content": "Question: " + question
        }
    ]

    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages
    )
    return response.choices[0].message.content
    

def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "ask_domain_expert",
                "description": "Ask a question to the domain expert",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to ask the domain expert"}
                    },
                    "required": ["question"],
                },
            },
        }
    ]