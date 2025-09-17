from openai import OpenAI
from dotenv import load_dotenv
import os
import traceback
import subprocess

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def web_search_stack_trace(query: str):
    """researches on how to fix the bug based on the stack trace and error message"""
    messages = [
            {
                "role": "user",
                "content": f"I am currently facing this bug: {query}. How do I fix it?"
            }
    ]

    response = client.responses.create(
        model="gpt-5",
        input=messages,
        tools=[
            {
                "type": "web_search"
            }
        ]
    )

    return response.output[0].content[0].text

def execute_code(filepath: str):
    try:
        # Run `python {filepath}` as a subprocess
        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return result.stdout
        else:
            trace = result.stderr
            search_result = web_search_stack_trace(trace)
            return trace + "\n" + search_result

    except Exception:
        trace = traceback.format_exc()
        search_result = web_search_stack_trace(trace)

        return trace + "\n" + search_result

if __name__ == "__main__":
    output = execute_code("code.py")
    print(output)