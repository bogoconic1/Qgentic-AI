import weave
import json
from openai import OpenAI
from tools.helpers import call_llm_with_retry  

from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

@weave.op()
def extract_fruit(sentence: str) -> dict:

    messages = [
        {
            "role": "user",
            "content": sentence
        }
    ]

    completion = call_llm_with_retry(
                client,
                model="openai/gpt-5:online",
                messages=messages,
            )
    try:
        extracted = completion.choices[0].message.content
    except Exception:
        extracted = ""
    return extracted

weave.init('intro-example')

sentence = "Who is Chris Deotte?"

extract_fruit(sentence)