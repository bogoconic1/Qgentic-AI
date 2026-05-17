"""Smoke test for GPT 5.5 via the OpenAI Responses API.

Tests three capabilities needed for the Gemini→OpenAI migration:
1. Basic response with reasoning={"effort": "high"}
2. Structured output via client.responses.parse(text_format=...)
3. Tool calling with function_call_output round-trip
"""

import json
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI()
MODEL = "gpt-5.5"
REASONING = {"effort": "xhigh"}


def test_basic_response():
    print(f"=== 1. Basic response (model={MODEL}, reasoning={REASONING}) ===")
    response = client.responses.create(
        model=MODEL,
        reasoning=REASONING,
        instructions="You are a concise assistant.",
        input="What is 97 * 83? Just the number.",
        temperature=1.0,
    )
    print(f"Text    : {response.output_text}")
    print(f"Tokens  : input={response.usage.input_tokens} output={response.usage.output_tokens}")
    print()


class CodeReview(BaseModel):
    has_bugs: bool
    severity: str
    explanation: str


def test_structured_output():
    print("=== 2. Structured output (responses.parse) ===")
    response = client.responses.parse(
        model=MODEL,
        reasoning=REASONING,
        instructions="Review the code snippet for bugs.",
        input="def avg(xs): return sum(xs) / len(xs)",
        text_format=CodeReview,
    )
    parsed = response.output_parsed
    print(f"Parsed  : {parsed}")
    print(f"Type    : {type(parsed).__name__}")
    print()


def test_tool_calling():
    print("=== 3. Tool calling with function_call_output ===")
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        {
            "type": "function",
            "name": "get_time",
            "description": "Get current time for a timezone.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        },
    ]

    response = client.responses.create(
        model=MODEL,
        reasoning=REASONING,
        input="What's the weather in Tokyo and the time in JST?",
        tools=tools,
    )

    function_calls = [item for item in response.output if item.type == "function_call"]
    print(f"Calls   : {len(function_calls)}")
    for fc in function_calls:
        print(f"  {fc.name}({fc.arguments}) call_id={fc.call_id}")

    tool_outputs = []
    for fc in function_calls:
        if fc.name == "get_weather":
            result = json.dumps({"temp_c": 22, "condition": "cloudy"})
        else:
            result = json.dumps({"time": "2026-05-17T14:30:00+09:00"})
        tool_outputs.append({
            "type": "function_call_output",
            "call_id": fc.call_id,
            "output": result,
        })

    follow_up = client.responses.create(
        model=MODEL,
        input=tool_outputs,
        previous_response_id=response.id,
        tools=tools,
    )
    print(f"Reply   : {follow_up.output_text}")
    print(f"Tokens  : input={follow_up.usage.input_tokens} output={follow_up.usage.output_tokens}")
    print()


if __name__ == "__main__":
    tests = [test_basic_response, test_structured_output, test_tool_calling]
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)
    print("All tests passed.")
