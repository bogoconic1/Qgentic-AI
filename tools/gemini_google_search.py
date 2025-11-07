import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class GeminiGoogleSearchClient:
    """Wrapper around Gemini for Google-enabled paper summarization."""

    _DEFAULT_PROMPT = (
        "The model name: {model_name}\n"
        "please provide the research paper summary for the model."
    )

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-pro") -> None:
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required to initialize GeminiGoogleSearchClient")

        self.client = genai.Client()
        self.model_name = model_name
        self._tools = [
            types.Tool(url_context=types.UrlContext()),
            types.Tool(googleSearch=types.GoogleSearch()),
        ]
        self._system_instruction = [
            types.Part.from_text(
                text=(
                    "You are an expert research analyst and summarization engine. \n\n"
                    "Firstly, you will be given a model name. You need to find the research paper url link "
                    "that is most relevant to the model via google search.\n\n"
                    "Second, after getting the url link (such as arxiv, acl, or semantic scholar), you need to "
                    "get the paper content and analyze provided research paper text and generate a structured, "
                    "section-by-section summary.\n\n"
                    "1. Format: output must be markdown.\n"
                    "2. Sections: You must include six sections, corresponding to the following sections: Abstract, "
                    "Introduction, Related Work, Method/Architecture, Experiments/Results, and Conclusion.\n"
                    "3. Summary Points: each section should contain a concise summary of 3 to 5 bullet points "
                    "(using Markdown list syntax *).\n"
                    "4. Content Focus:\n"
                    "- Method/Architecture: Focus on technical concepts, novel components, and parameters.\n"
                    "- Experiments/Results: Focus on quantitative data, benchmark scores, comparative gains, and "
                    "key ablation findings.\n"
                    "- Purpose: it must contain a brief, descriptive sentence explaining the section's function in "
                    "the paper."
                ),
            ),
        ]

    def generate_summary(self, model_name: str, user_prompt: str | None = None) -> str:
        prompt = (user_prompt or self._DEFAULT_PROMPT).format(
            model_name=model_name)
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        config = types.GenerateContentConfig(
            temperature=0.3,
            tools=self._tools,
            system_instruction=self._system_instruction,
        )

        output_chunks: list[str] = []
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                output_chunks.append(chunk.text)

        return "".join(output_chunks)
