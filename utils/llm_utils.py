"""
LLM utility functions for provider detection and response handling.
"""

import base64
import logging
import mimetypes
from pathlib import Path

logger = logging.getLogger(__name__)


def encode_image_to_data_url(image_path: str | Path, resize_for_anthropic: bool = False) -> str:
    """
    Encode an image file to a data URL.

    Args:
        image_path: Path to the image file
        resize_for_anthropic: Whether to resize image to max 2000px per dimension (for Anthropic API)

    Returns:
        Data URL string in format: data:{mime};base64,{data}
        Empty string if encoding fails

    Examples:
        >>> encode_image_to_data_url("chart.png")
        'data:image/png;base64,iVBORw0KGgo...'

        >>> encode_image_to_data_url("large_chart.jpg", resize_for_anthropic=True)
        'data:image/jpeg;base64,/9j/4AAQSkZJRg...'
    """
    path = Path(image_path) if isinstance(image_path, str) else image_path

    # Determine MIME type
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        elif suffix == ".gif":
            mime = "image/gif"
        else:
            return ""

    try:
        image_bytes = path.read_bytes()

        # Resize image if needed for Anthropic (max 2000px per dimension for multi-image requests)
        if resize_for_anthropic:
            try:
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size

                # Check if resizing is needed
                if width > 2000 or height > 2000:
                    # Calculate new dimensions maintaining aspect ratio
                    if width > height:
                        new_width = 2000
                        new_height = int(height * (2000 / width))
                    else:
                        new_height = 2000
                        new_width = int(width * (2000 / height))

                    # Resize the image
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Save to bytes
                    output = io.BytesIO()
                    # Preserve format, default to PNG if format is unknown
                    img_format = img.format if img.format else 'PNG'
                    img.save(output, format=img_format)
                    image_bytes = output.getvalue()

                    logger.info(f"Resized image {path.name} from {width}x{height} to {new_width}x{new_height} for Anthropic API")
            except ImportError:
                logger.warning("PIL not available for image resizing. Image may fail if dimensions exceed 2000px.")
            except Exception as e:
                logger.warning(f"Failed to resize image {path.name}: {e}. Using original.")

        data = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {path}: {e}")
        return ""

    return f"data:{mime};base64,{data}"


def detect_provider(model_name: str) -> str:
    """
    Detect LLM provider from model name.

    Args:
        model_name: Model identifier (e.g., "gpt-5", "claude-sonnet-4-5-20250929", "gemini-2.5-pro")

    Returns:
        "openai" | "anthropic" | "gemini"

    Raises:
        ValueError: If provider cannot be determined

    Examples:
        >>> detect_provider("gpt-5")
        'openai'
        >>> detect_provider("claude-sonnet-4-5-20250929")
        'anthropic'
        >>> detect_provider("gemini-2.5-pro")
        'gemini'
    """
    if model_name.startswith("gpt-") or model_name.startswith("o"):
        return "openai"
    elif model_name.startswith("claude-"):
        return "anthropic"
    elif model_name.startswith("gemini-"):
        return "google"
    else:
        raise ValueError(f"Unknown provider for model: {model_name}")


def extract_text_from_response(response, provider: str) -> str:
    """
    Extract text from provider-specific response format.

    Args:
        response: OpenAI, Anthropic, or Google response object
        provider: "openai", "anthropic", or "google"

    Returns:
        Extracted text string

    Raises:
        ValueError: If provider is not supported

    Examples:
        # OpenAI response
        >>> text = extract_text_from_response(openai_response, "openai")

        # Anthropic response
        >>> text = extract_text_from_response(anthropic_response, "anthropic")

        # Google/Gemini response
        >>> text = extract_text_from_response(gemini_response, "google")
    """
    if provider == "openai":
        return response.output_text

    elif provider == "anthropic":
        # Concatenate all text blocks from content
        text_blocks = [
            block.text for block in response.content
            if hasattr(block, 'text')
        ]
        return ''.join(text_blocks)

    elif provider == "google":
        # For Gemini, response.text contains the generated text
        return response.text if hasattr(response, 'text') else str(response)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

def append_message(provider: str, role: str, message: str) -> dict:
    """
    Create a message in provider-specific format.

    Args:
        provider: "openai", "anthropic", or "google"
        role: Message role ("user", "assistant", "model", etc.)
        message: Message content (text string)

    Returns:
        Formatted message dict for the provider

    Examples:
        >>> append_message("openai", "user", "Hello")
        {'role': 'user', 'content': 'Hello'}

        >>> append_message("google", "user", "Hello")
        {'role': 'user', 'parts': [{'text': 'Hello'}]}

        >>> append_message("google", "assistant", "Hi")
        {'role': 'model', 'parts': [{'text': 'Hi'}]}
    """
    if provider == "google":
        # Gemini uses "model" instead of "assistant"
        gemini_role = "model" if role == "assistant" else role
        return {"role": gemini_role, "parts": [{"text": message}]}
    else:
        # OpenAI and Anthropic use same format
        return {"role": role, "content": message}


def get_tools_openai(max_parallel_workers: int = 1):
    """
    Get tools in OpenAI format.

    Args:
        max_parallel_workers: Maximum parallel workers for run_ab_test

    Returns:
        List of tool definitions in OpenAI format
    """
    return [
        {
            "type": "function",
            "name": "ask_eda",
            "description": "Ask a question to the EDA expert",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the EDA expert"}
                },
            },
            "additionalProperties": False,
            "required": ['question']
        },
        {
            "type": "function",
            "name": "run_ab_test",
            "description": f"Run A/B tests to validate modeling or feature engineering choices by comparing their impact on performance. You can ask up to {max_parallel_workers} questions in parallel for efficiency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": f"List of A/B testing questions to run in parallel (max {max_parallel_workers}). Each question should be a comparison test",
                        "items": {"type": "string"},
                    }
                },
            },
            "additionalProperties": False,
            "required": ['questions']
        },
        {
            "type": "function",
            "name": "download_external_datasets",
            "description": "Download external data to working directory by searching with 3 different phrasings to maximize search coverage",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_1": {"type": "string", "description": "First phrasing of the dataset query"},
                    "question_2": {"type": "string", "description": "Second phrasing with different wording"},
                    "question_3": {"type": "string", "description": "Third phrasing using alternative keywords"}
                },
            },
            "additionalProperties": False,
            "required": ["question_1", "question_2", "question_3"],
        },
        {
            "type": "function",
            "name": "read_research_paper",
            "description": "Read and summarize a research paper from arxiv. Returns structured markdown summary with Abstract, Introduction, Related Work, Method/Architecture, Experiments/Results, and Conclusion sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_link": {"type": "string", "description": "ArXiv paper link (e.g., 'https://arxiv.org/pdf/2510.22916' or just '2510.22916')"}
                },
            },
            "additionalProperties": False,
            "required": ["arxiv_link"],
        },
        {
            "type": "function",
            "name": "scrape_web_page",
            "description": "Scrape a web page and return markdown content. Useful for reading blog posts, documentation, technical tutorials, and other domain-specific web content that complements arxiv papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The webpage URL to scrape (e.g., 'https://developer.nvidia.com/blog/...')"}
                },
            },
            "additionalProperties": False,
            "required": ["url"],
        },
    ]


def get_tools_anthropic(max_parallel_workers: int = 1):
    """
    Get tools in Anthropic format.

    Key differences from OpenAI format:
    - No nested "function" wrapper
    - Uses "input_schema" instead of "parameters"
    - 3-4 sentence descriptions (Anthropic best practice)

    Args:
        max_parallel_workers: Maximum parallel workers for run_ab_test

    Returns:
        List of tool definitions in Anthropic format
    """
    return [
        {
            "name": "ask_eda",
            "description": """Ask a question to the EDA expert for exploratory data analysis.
            Returns statistical insights, data quality assessments, and feature analysis based on
            the provided dataset. Use this tool when you need to understand dataset characteristics,
            distributions, correlations, or potential data issues before deciding on modeling approaches.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the EDA expert"
                    }
                },
                "required": ["question"]
            }
        },
        {
            "name": "run_ab_test",
            "description": f"""Run A/B tests to validate modeling or feature engineering choices by comparing
            their impact on performance. You can ask up to {max_parallel_workers} questions in parallel for efficiency.
            Returns performance comparisons with statistical significance. Use this when you need empirical
            evidence to choose between different approaches or validate hypotheses about model improvements.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": f"List of A/B testing questions to run in parallel (max {max_parallel_workers}). Each question should be a comparison test",
                        "items": {"type": "string"}
                    }
                },
                "required": ["questions"]
            }
        },
        {
            "name": "download_external_datasets",
            "description": """Download external datasets to augment your training data by searching with
            3 different phrasings to maximize search coverage. Retrieves publicly available datasets from
            Kaggle and other sources. Use this when the current dataset is insufficient or you need additional
            examples for specific classes, features, or data diversity to improve model generalization.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question_1": {
                        "type": "string",
                        "description": "First phrasing of the dataset query"
                    },
                    "question_2": {
                        "type": "string",
                        "description": "Second phrasing with different wording"
                    },
                    "question_3": {
                        "type": "string",
                        "description": "Third phrasing using alternative keywords"
                    }
                },
                "required": ["question_1", "question_2", "question_3"]
            }
        },
        {
            "name": "read_research_paper",
            "description": """Read and summarize academic research papers from arXiv relevant to the machine learning task.
            Returns structured markdown summary with Abstract, Introduction, Related Work, Method/Architecture,
            Experiments/Results, and Conclusion sections. Use this to discover state-of-the-art techniques,
            validate your approach against published research, or find novel methods applicable to your problem.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "arxiv_link": {
                        "type": "string",
                        "description": "ArXiv paper link (e.g., 'https://arxiv.org/pdf/2510.22916' or just '2510.22916')"
                    }
                },
                "required": ["arxiv_link"]
            }
        },
        {
            "name": "scrape_web_page",
            "description": """Scrape web pages and return markdown content from technical documentation, blog posts,
            and tutorials. Useful for accessing domain-specific knowledge, implementation guides, or best practices
            that complement academic papers. Use this to gather practical insights from developer blogs, official
            documentation, or technical articles that can inform your modeling approach.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The webpage URL to scrape (e.g., 'https://developer.nvidia.com/blog/...')"
                    }
                },
                "required": ["url"]
            }
        }
    ]


def get_tools_gemini(max_parallel_workers: int = 1):
    """
    Get tools in Gemini format (FunctionDeclaration).

    Args:
        max_parallel_workers: Maximum parallel workers for run_ab_test

    Returns:
        List of FunctionDeclaration objects for Gemini
    """
    from google.genai import types

    return [
        types.FunctionDeclaration(
            name="ask_eda",
            description="Ask a question to the EDA expert for exploratory data analysis. Returns statistical insights, data quality assessments, and feature analysis based on the provided dataset.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the EDA expert"
                    }
                },
                "required": ["question"]
            }
        ),
        types.FunctionDeclaration(
            name="run_ab_test",
            description=f"Run A/B tests to validate modeling or feature engineering choices by comparing their impact on performance. You can ask up to {max_parallel_workers} questions in parallel for efficiency.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": f"List of A/B testing questions to run in parallel (max {max_parallel_workers}). Each question should be a comparison test",
                        "items": {"type": "string"}
                    }
                },
                "required": ["questions"]
            }
        ),
        types.FunctionDeclaration(
            name="download_external_datasets",
            description="Download external datasets to augment your training data by searching with 3 different phrasings to maximize search coverage. Retrieves publicly available datasets from Kaggle and other sources.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "question_1": {
                        "type": "string",
                        "description": "First phrasing of the dataset query"
                    },
                    "question_2": {
                        "type": "string",
                        "description": "Second phrasing with different wording"
                    },
                    "question_3": {
                        "type": "string",
                        "description": "Third phrasing using alternative keywords"
                    }
                },
                "required": ["question_1", "question_2", "question_3"]
            }
        ),
        types.FunctionDeclaration(
            name="read_research_paper",
            description="Read and summarize academic research papers from arXiv relevant to the machine learning task. Returns structured markdown summary with Abstract, Introduction, Related Work, Method/Architecture, Experiments/Results, and Conclusion sections.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "arxiv_link": {
                        "type": "string",
                        "description": "ArXiv paper link (e.g., 'https://arxiv.org/pdf/2510.22916' or just '2510.22916')"
                    }
                },
                "required": ["arxiv_link"]
            }
        ),
        types.FunctionDeclaration(
            name="scrape_web_page",
            description="Scrape web pages and return markdown content from technical documentation, blog posts, and tutorials. Useful for accessing domain-specific knowledge, implementation guides, or best practices that complement academic papers.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The webpage URL to scrape (e.g., 'https://developer.nvidia.com/blog/...')"
                    }
                },
                "required": ["url"]
            }
        )
    ]


def get_tools_for_provider(provider: str, max_parallel_workers: int = 1):
    """
    Get tools in the appropriate format for the provider.

    Args:
        provider: "openai", "anthropic", or "google"
        max_parallel_workers: Maximum parallel workers for run_ab_test

    Returns:
        List of tool definitions in provider-specific format

    Raises:
        ValueError: If provider is not supported
    """
    if provider == "openai":
        return get_tools_openai(max_parallel_workers)
    elif provider == "anthropic":
        return get_tools_anthropic(max_parallel_workers)
    elif provider == "google":
        return get_tools_gemini(max_parallel_workers)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
