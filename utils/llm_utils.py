"""
LLM utility functions for provider detection and response handling.
"""

import base64
import logging
import mimetypes
from pathlib import Path

logger = logging.getLogger(__name__)


def encode_image_to_data_url(image_path: str | Path, resize_for_anthropic: bool = False, max_size_bytes: int = 4_500_000) -> str:
    """
    Encode an image file to a data URL.

    Args:
        image_path: Path to the image file
        resize_for_anthropic: Whether to resize image to max 2000px per dimension (for Anthropic API)
        max_size_bytes: Maximum file size in bytes (default 4.5MB to stay under Anthropic's 5MB limit)

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

        # Resize/compress image if needed for Anthropic
        if resize_for_anthropic:
            try:
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size
                original_size = len(image_bytes)
                resized = False

                # Check if resizing is needed (dimensions)
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
                    resized = True
                    logger.info(f"Resized image {path.name} from {width}x{height} to {new_width}x{new_height}")

                # Save to bytes (convert PNG to JPEG if too large)
                output = io.BytesIO()

                # Try saving as original format first
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB for JPEG
                    img = img.convert('RGB')

                # Save as JPEG with quality adjustment to meet size limit
                quality = 95
                while quality >= 20:
                    output = io.BytesIO()
                    img.save(output, format='JPEG', quality=quality, optimize=True)
                    image_bytes = output.getvalue()

                    if len(image_bytes) <= max_size_bytes:
                        if quality < 95 or resized:
                            logger.info(f"Compressed image {path.name}: {original_size/1024/1024:.2f}MB -> {len(image_bytes)/1024/1024:.2f}MB (quality={quality})")
                        mime = "image/jpeg"  # Update mime type since we converted to JPEG
                        break
                    quality -= 10
                else:
                    # Still too large after max compression, resize further
                    current_width, current_height = img.size
                    new_width = int(current_width * 0.5)
                    new_height = int(current_height * 0.5)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    output = io.BytesIO()
                    img.save(output, format='JPEG', quality=50, optimize=True)
                    image_bytes = output.getvalue()
                    mime = "image/jpeg"
                    logger.warning(f"Aggressively compressed image {path.name} to {len(image_bytes)/1024/1024:.2f}MB")

            except ImportError:
                logger.warning("PIL not available for image resizing. Image may fail if too large.")
            except Exception as e:
                logger.warning(f"Failed to resize image {path.name}: {e}. Using original.")

        # Final size check
        if len(image_bytes) > max_size_bytes:
            logger.error(f"Image {path.name} exceeds size limit ({len(image_bytes)/1024/1024:.2f}MB > {max_size_bytes/1024/1024:.2f}MB). Skipping.")
            return ""

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


def get_tools_openai():
    """
    Get tools in OpenAI format.

    Returns:
        List of tool definitions in OpenAI format
    """
    return [
        {
            "type": "function",
            "name": "execute_python",
            "description": "Write and execute a Python script. The script runs in the task data directory with access to all data files, model outputs, and predictions. Print results to stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Complete Python script to execute."}
                },
            },
            "additionalProperties": False,
            "required": ['code']
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


def get_tools_anthropic():
    """
    Get tools in Anthropic format.

    Key differences from OpenAI format:
    - No nested "function" wrapper
    - Uses "input_schema" instead of "parameters"
    - 3-4 sentence descriptions (Anthropic best practice)

    Returns:
        List of tool definitions in Anthropic format
    """
    return [
        {
            "name": "execute_python",
            "description": """Write and execute a Python script. The script runs in the task data directory
            with access to all data files, model outputs, and predictions. Use this for exploratory data analysis,
            debugging, downloading external datasets, or any data investigation. Print results to stdout.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python script to execute."
                    }
                },
                "required": ["code"]
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


def get_tools_gemini():
    """
    Get tools in Gemini format (FunctionDeclaration).

    Returns:
        List of FunctionDeclaration objects for Gemini
    """
    from google.genai import types

    return [
        types.FunctionDeclaration(
            name="execute_python",
            description="Write and execute a Python script. The script runs in the task data directory with access to all data files, model outputs, and predictions. Print results to stdout.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python script to execute."
                    }
                },
                "required": ["code"]
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


def get_tools_for_provider(provider: str):
    """
    Get tools in the appropriate format for the provider.

    Args:
        provider: "openai", "anthropic", or "google"

    Returns:
        List of tool definitions in provider-specific format

    Raises:
        ValueError: If provider is not supported
    """
    if provider == "openai":
        return get_tools_openai()
    elif provider == "anthropic":
        return get_tools_anthropic()
    elif provider == "google":
        return get_tools_gemini()
    else:
        raise ValueError(f"Unsupported provider: {provider}")
