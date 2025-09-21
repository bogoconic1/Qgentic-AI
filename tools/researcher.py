from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
import re
from io import StringIO

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")


def ask_eda(question: str, description: str, data_path: str, max_attempts: int = 5) -> str:
    """Asks a question about the data provided for exploratory data analysis (EDA)"""
    PROMPT = f"""You are an experienced Kaggle Competitions Grandmaster. Your goal is to write code that answers questions about the data provided.
Competition Description:
{description}

You will be given one or more questions related to the data. Your task is to generate code that, when executed, will answer all questions using the data provided.
Before generating the code, provide around 5 lines of reasoning about the approach.

The data files are stored in the directory "{data_path}".

So make sure you put your code within a python code block like this:
```python
data_path = "{data_path}"
<YOUR CODE HERE>
```

IMPORTANT: Always provide descriptive answers. Instead of just printing a number like "100", print a complete sentence like "There are a total of 100 records". Make your final answer clear and informative.
"""
    all_messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": question},
    ]

    pattern = r'```python\s*(.*?)\s*```'

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        print(f"Attempt {attempt} to answer the question.")
        completion = client.chat.completions.create(
            extra_body={},
            model="qwen/qwen3-coder",
            messages=all_messages
        )
        response_text = completion.choices[0].message.content or ""

        matches = re.findall(pattern, response_text, re.DOTALL)
        code = "\n\n".join(matches).strip()
        print(code)

        if not code:
            last_error = "No python code block found in the model response."
        else:
            try:
                # Simple policy assertion carried forward
                assert "train_labels.csv" not in code or "!= 'Missing'" in code, (
                    "You must remove records with type = 'Missing' before doing any analysis."
                )

                # Save current stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                exec(code)
                output = captured_output.getvalue()

                sys.stdout = old_stdout

                return output

            except AssertionError as e:
                try:
                    sys.stdout = old_stdout
                except Exception:
                    pass
                last_error = f"Assertion failed: {e}"
                print(last_error)
            except Exception as e:
                try:
                    sys.stdout = old_stdout
                except Exception:
                    pass
                last_error = f"Error executing code: {str(e)}"
                print(last_error)

    return "Your question cannot be answered."
    

def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "ask_eda",
                "description": "Ask a question to the EDA expert",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to ask the EDA expert"}
                    },
                    "required": ["question"],
                },
            },
        }
    ]