import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-flash-latest"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    tools = [
        types.Tool(url_context=types.UrlContext()),
        types.Tool(googleSearch=types.GoogleSearch(
        )),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
        image_config=types.ImageConfig(
            image_size="1K",
        ),
        tools=tools,
        system_instruction=[
            types.Part.from_text(text="""You are an expert research analyst and summarization engine. 

Firstly, you will be given a model name. You need to find the research paper url link that is most relevant to the model via google search.

Second, after getting the url link (such as arxiv, acl, or semantic scholar), you need to get the paper content and analyze provided research paper text and generate a structured, section-by-section summary.

1. Format: output must be markdown.
2. Sections: You must include six sections, corresponding to the following sections: Abstract, Introduction, Related Work, Method/Architecture, Experiments/Results, and Conclusion.
3. Summary Points: each section should contain a concise summary of 3 to 5 bullet points (using Markdown list syntax *).
4. Content Focus:
- Method/Architecture: Focus on technical concepts, novel components, and parameters.
- Experiments/Results: Focus on quantitative data, benchmark scores, comparative gains, and key ablation findings.
- Purpose: it must contain a brief, descriptive sentence explaining the section's function in the paper."""),
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")


if __name__ == "__main__":
    generate()
