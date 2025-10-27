"""
Tools for enhanced developer agent: summary generation and cross-model enhancement analysis.

This module provides functions for:
1. Generating technical summaries of baseline model implementations
2. Analyzing cross-model learnings to generate enhancement recommendations
"""

from pathlib import Path
from typing import Dict, List
import yaml

from prompts.ensembler_agent import (
    ensembler_code_summarize_system_prompt,
    ensembler_code_summarize_user_prompt,
    ensembler_code_enhance_system_prompt,
    ensembler_code_enhance_user_prompt
)
from tools.helpers import call_llm_with_retry

# Load config
with open("config.yaml", "r") as f:
    _CONFIG = yaml.safe_load(f)
_LLM_CFG = _CONFIG.get("llm")


def generate_model_summary(
    description: str,
    model_name: str,
    code: str,
    logs: str
) -> str:
    """
    Generate a technical summary of a model's implementation.

    Args:
        description: Competition description
        model_name: Model name (e.g., "DeBERTa-v3-large")
        code: Model implementation code
        logs: Execution logs

    Returns:
        Technical summary markdown string
    """
    system_prompt = ensembler_code_summarize_system_prompt()
    user_prompt = ensembler_code_summarize_user_prompt(description, model_name, code, logs)

    model = _LLM_CFG["developer_tool_model"]
    response = call_llm_with_retry(
        model=model,
        instructions=system_prompt,
        tools=[],
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.output_text


def generate_enhancement_recommendations(
    target_model_name: str,
    target_summary: str,
    target_successful_ideas: List[str],
    target_blacklisted_ideas: List[str],
    other_models: List[Dict]
) -> str:
    """
    Generate enhancement recommendations for a target model based on other models' learnings.

    Args:
        target_model_name: Name of the target model to enhance
        target_summary: Technical summary of target model
        target_successful_ideas: List of successful ideas from target model
        target_blacklisted_ideas: List of blacklisted ideas from target model
        other_models: List of dicts with keys: name, summary, successful_ideas, blacklisted_ideas

    Returns:
        Enhancement recommendations markdown string
    """
    system_prompt = ensembler_code_enhance_system_prompt()
    user_prompt = ensembler_code_enhance_user_prompt(
        target_model_name=target_model_name,
        target_summary=target_summary,
        target_successful_ideas=target_successful_ideas,
        target_blacklisted_ideas=target_blacklisted_ideas,
        other_models=other_models
    )

    model = _LLM_CFG["developer_tool_model"]
    response = call_llm_with_retry(
        model=model,
        instructions=system_prompt,
        tools=[],
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.output_text


def generate_all_summaries(
    competition_description_path: Path,
    baseline_results: Dict,
    outputs_dir: Path,
    iteration: int
):
    """
    Generate technical summaries for all baseline models.

    Args:
        competition_description_path: Path to description.md
        baseline_results: Dict of baseline results (from baseline_results.json)
        outputs_dir: Path to outputs directory
        iteration: Iteration number

    Returns:
        Dict mapping model_name to summary markdown
    """
    with open(competition_description_path, 'r') as f:
        description = f.read()

    summaries = {}
    ensemble_dir = outputs_dir / str(iteration) / "ensemble"
    summaries_dir = ensemble_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    for idx, (model_name, baseline_info) in enumerate(baseline_results.items(), start=1):
        print(f"Generating summary for Model {idx}: {model_name}")

        # Read code file
        best_code_file = baseline_info.get("best_code_file")
        if not best_code_file:
            print(f"  WARNING: No code file for {model_name}, skipping")
            continue

        code_path = outputs_dir / str(iteration) / best_code_file
        if not code_path.exists():
            print(f"  WARNING: Code file not found: {code_path}, skipping")
            continue

        with open(code_path, 'r') as f:
            code = f.read()

        # Read logs file
        log_file = best_code_file.replace('.py', '.txt')
        log_path = outputs_dir / str(iteration) / log_file
        if not log_path.exists():
            print(f"  WARNING: Log file not found: {log_path}, using empty logs")
            logs = ""
        else:
            with open(log_path, 'r') as f:
                logs = f.read()

        # Generate summary
        summary = generate_model_summary(
            description=description,
            model_name=model_name,
            code=code,
            logs=logs
        )

        # Save summary
        summary_path = summaries_dir / f"summary_model_{idx}.md"
        with open(summary_path, 'w') as f:
            f.write(summary)

        summaries[model_name] = summary
        print(f"  Saved summary to {summary_path}")

    return summaries


def generate_all_enhancements(
    baseline_results: Dict,
    summaries: Dict[str, str],
    outputs_dir: Path,
    iteration: int
):
    """
    Generate enhancement recommendations for all models based on cross-model analysis.

    Args:
        baseline_results: Dict of baseline results (from baseline_results.json)
        summaries: Dict mapping model_name to summary markdown
        outputs_dir: Path to outputs directory
        iteration: Iteration number

    Returns:
        Dict mapping model_name to enhancement recommendations markdown
    """
    enhancements = {}
    ensemble_dir = outputs_dir / str(iteration) / "ensemble"
    enhancements_dir = ensemble_dir / "enhancements"
    enhancements_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(baseline_results.keys())

    for idx, target_model_name in enumerate(model_names, start=1):
        print(f"Generating enhancements for Model {idx}: {target_model_name}")

        target_info = baseline_results[target_model_name]
        target_summary = summaries.get(target_model_name)

        if not target_summary:
            print(f"  WARNING: No summary for {target_model_name}, skipping")
            continue

        # Build other models list (all except target)
        other_models = []
        for other_idx, other_model_name in enumerate(model_names, start=1):
            if other_model_name == target_model_name:
                continue

            other_info = baseline_results[other_model_name]
            other_summary = summaries.get(other_model_name)

            if not other_summary:
                print(f"  WARNING: No summary for {other_model_name}, skipping in cross-analysis")
                continue

            other_models.append({
                "name": other_model_name,
                "summary": other_summary,
                "successful_ideas": other_info.get("successful_ideas", []),
                "blacklisted_ideas": other_info.get("blacklisted_ideas", [])
            })

        if not other_models:
            print(f"  WARNING: No other models to analyze, skipping")
            continue

        # Generate enhancement recommendations
        enhancement = generate_enhancement_recommendations(
            target_model_name=target_model_name,
            target_summary=target_summary,
            target_successful_ideas=target_info.get("successful_ideas", []),
            target_blacklisted_ideas=target_info.get("blacklisted_ideas", []),
            other_models=other_models
        )

        # Save enhancement
        enhancement_path = enhancements_dir / f"enhancements_model_{idx}.md"
        with open(enhancement_path, 'w') as f:
            f.write(enhancement)

        enhancements[target_model_name] = enhancement
        print(f"  Saved enhancements to {enhancement_path}")
