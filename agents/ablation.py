import logging
import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config
from tools.helpers import call_llm_with_retry
import weave


logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm", {}) if isinstance(_CONFIG, dict) else {}
_BASE_URL = _LLM_CFG.get("base_url", "https://openrouter.ai/api/v1")
_API_KEY_ENV = _LLM_CFG.get("api_key_env", "OPENROUTER_API_KEY")


class AblationAgent:
    def __init__(self) -> None:
        load_dotenv()
        self.client = OpenAI(api_key=os.environ.get(_API_KEY_ENV), base_url=_BASE_URL)

    @weave.op()
    def summarize_baseline(self, baseline: Dict) -> Dict[str, str]:
        """Return a short natural-language summary for the baseline run.

        Expected baseline keys: score (float), version (int), guardrail_notes (str),
        cv_lb_gap (str|float) optional, key_findings (list[str]) optional.
        """
        summary_prompt = (
            "You will receive a baseline result for a Kaggle pipeline.\n"
            "Write a concise summary (<= 180 words) covering: score achieved, any\n"
            "guardrail or stability notes, and 3-5 bullet takeaways on strengths/gaps.\n"
            "Return plain text only.\n\n"
            f"<baseline>\n{baseline}\n</baseline>"
        )
        try:
            completion = call_llm_with_retry(
                self.client,
                model=_LLM_CFG.get("developer_tool_model", _LLM_CFG.get("developer_model", "openai/gpt-5")),
                messages=[{"role": "user", "content": summary_prompt}],
            )
            msg = completion.choices[0].message
            content = msg.content or ""
        except Exception:
            logger.exception("Baseline ablation summarization failed")
            content = "Baseline summary unavailable."
        return {"summary": content.strip()}

    @weave.op()
    def summarize_batch(self, baseline: Dict, batch_results: List[Dict]) -> Dict[str, str]:
        """Summarize a batch of four suggestions with their scores/log notes into one paragraph.

        Return a dict with a single key {"summary": str} suitable for steering SOTA next.\n
        """
        prompt = (
            "You will receive a baseline snapshot and four candidate runs with their\n"
            "scores/status and short notes. Produce a compact ablation summary (<= 220\n"
            "words) stating: baseline score, each candidate's score, which helped or\n"
            "hurt, and one-line guidance for what to pursue next. Return plain text only.\n\n"
            f"<baseline>\n{baseline}\n</baseline>\n\n"
            f"<batch_results>\n{batch_results}\n</batch_results>"
        )
        try:
            completion = call_llm_with_retry(
                self.client,
                model=_LLM_CFG.get("developer_tool_model", _LLM_CFG.get("developer_model", "openai/gpt-5")),
                messages=[{"role": "user", "content": prompt}],
            )
            msg = completion.choices[0].message
            content = msg.content or ""
        except Exception:
            logger.exception("Batch ablation summarization failed")
            content = "Batch summary unavailable."
        return {"summary": content.strip()}


