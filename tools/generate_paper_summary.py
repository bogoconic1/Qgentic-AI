import os
from dotenv import load_dotenv
import weave

from project_config import get_config
from tools.helpers import call_llm_with_retry, call_llm_with_retry_anthropic, call_llm_with_retry_google
from utils.llm_utils import detect_provider, extract_text_from_response, append_message

load_dotenv()


class PaperSummaryClient:
    """Wrapper for paper summarization using multi-provider LLM support."""

    _DEFAULT_PROMPT_MODEL = (
        "The model name: {model_name}\n"
        "please provide the research paper summary for the model."
    )

    _DEFAULT_PROMPT_ARXIV = (
        "The arXiv paper ID is: {model_name}\n"
        "Access the paper at: https://arxiv.org/abs/{model_name}\n"
        "Please provide a research paper summary."
    )

    def __init__(self, api_key: str | None = None, is_model: bool = True) -> None:
        cfg = get_config()
        self.model_name = cfg["llm"]["paper_summary_model"]
        self.is_model = is_model

        # Detect provider from model name
        self.provider = detect_provider(self.model_name)

        if self.is_model:
            self.system_instruction = (
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
            )
        else:
            self.system_instruction = (
                "You are an expert research analyst and summarization engine. \n\n"
                "Firstly, you will be given an **arXiv paper ID** (e.g., 1708.24371).\n\n"
                "Second, you need to **access the paper's content** using this ID (e.g., by retrieving text from `https://arxiv.org/abs/<id>` or `https://arxiv.org/pdf/<id>`) "
                "and then analyze the retrieved research paper text to generate a structured, "
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
            )

    @weave.op()
    def generate_summary(self, model_name: str, user_prompt: str | None = None) -> str:
        if user_prompt:
            prompt = user_prompt.format(model_name=model_name)
        else:
            default_prompt = self._DEFAULT_PROMPT_MODEL if self.is_model else self._DEFAULT_PROMPT_ARXIV
            prompt = default_prompt.format(model_name=model_name)

        # Build messages in provider-specific format
        messages = [append_message(self.provider, "user", prompt)]

        # Call appropriate LLM based on provider
        if self.provider == "openai":
            response = call_llm_with_retry(
                model=self.model_name,
                instructions=self.system_instruction,
                tools=[],
                messages=messages,
                web_search_enabled=True,
            )
            result = extract_text_from_response(response, "openai")
        elif self.provider == "anthropic":
            response = call_llm_with_retry_anthropic(
                model=self.model_name,
                instructions=self.system_instruction,
                tools=[],
                messages=messages,
                web_search_enabled=True,
            )
            result = extract_text_from_response(response, "anthropic")
        elif self.provider == "google":
            response = call_llm_with_retry_google(
                model=self.model_name,
                system_instruction=self.system_instruction,
                messages=messages,
                enable_google_search=True,
            )
            # Extract text from Gemini response
            result = response.text if hasattr(response, 'text') else str(response)
        else:
            return f"Error: Unsupported provider '{self.provider}'"

        return result if result else "Error: Failed to generate summary"
