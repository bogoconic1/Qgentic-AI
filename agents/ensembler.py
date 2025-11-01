import logging
import json
from pathlib import Path
from typing import Optional

from agents.developer import DeveloperAgent
from utils.code_utils import strip_header_from_code
from prompts.ensembler_agent import build_system as prompt_build_system


class EnsemblerAgent(DeveloperAgent):
    """
    Ensemble specialist agent that combines multiple baseline models.

    Inherits from DeveloperAgent and follows the same flow, with differences:
    - Uses ensemble-specific system prompt
    - Uses ensemble strategies instead of SOTA suggestions
    - Operates in parallel with one agent per strategy
    - allow_multi_fold is always True (ensembles benefit from multi-fold approaches)
    - Files named: code_{iteration}_{strategy_index}_ens_v{version}.py
    - Logs named: code_{iteration}_{strategy_index}_ens_v{version}.txt
    - Submissions named: submission_{version}.csv
    """

    def __init__(
        self,
        slug: str,
        iteration: int,
        strategy_index: int,
        strategy: dict,
        baseline_metadata: dict,
        cpu_core_range: Optional[list[int]] = None,
        gpu_identifier: Optional[str] = None,
        gpu_isolation_mode: str = "none",
        conda_env: Optional[str] = None,
    ):
        """
        Initialize EnsemblerAgent for a single ensemble strategy.

        Args:
            slug: Competition slug
            iteration: Iteration number
            strategy_index: Index of this strategy (1-8)
            strategy: Single strategy dict with "strategy" and "models_needed" keys
            baseline_metadata: Metadata from ensemble_metadata.json
            cpu_core_range: CPU cores to use
            gpu_identifier: GPU identifier for isolation
            gpu_isolation_mode: GPU isolation mode
            conda_env: Conda environment name
        """
        self.strategy_index = strategy_index
        self.strategy = strategy
        self.baseline_metadata = baseline_metadata

        # Extract strategy text for model_name
        strategy_text = strategy.get("strategy", "")
        strategy_name = f"EnsembleStrategy{strategy_index}"

        # Pass modified iteration to parent for automatic file naming
        # e.g., iteration="5_1_ens" produces code_5_1_ens_v1.py automatically
        iteration_ens = f"{iteration}_{strategy_index}_ens"

        # Initialize parent with ensemble-specific iteration
        super().__init__(
            slug=slug,
            iteration=iteration_ens,
            model_name=strategy_name,
            model_recommendations=strategy_text,  # Pass strategy as recommendations
            later_recommendations=None,
            cpu_core_range=cpu_core_range,
            gpu_identifier=gpu_identifier,
            gpu_isolation_mode=gpu_isolation_mode,
            conda_env=conda_env,
        )

        # Store original iteration for reference
        self.original_iteration = iteration

        # Locate ensemble folder with baseline outputs
        self.ensemble_folder = self.base_dir / self.outputs_dirname / str(iteration) / "ensemble"

        self.logger.info(
            "Initialized EnsemblerAgent for strategy %d: %s",
            strategy_index,
            strategy_text[:100] + "..." if len(strategy_text) > 100 else strategy_text
        )

    def _compose_system(self, allow_multi_fold: bool = False) -> str:
        """
        Override to use ensemble-specific system prompt.

        Note: allow_multi_fold parameter is ignored for ensembles (always True),
        but is accepted for compatibility with parent class signature.

        Returns:
            System prompt for ensemble agent
        """
        self.logger.debug("Composing ensemble system prompt (multi-fold always enabled for ensembles)")

        # Load baseline code files based on models_needed for this strategy
        file_contents = {}
        models_needed = self.strategy.get("models_needed", [])

        for model_name in models_needed:
            # Find the code file for this model in baseline_metadata
            model_data = self.baseline_metadata.get(model_name, {})
            code_file = model_data.get("best_code_file")

            if code_file:
                code_path = self.ensemble_folder / code_file
                if code_path.exists():
                    # Strip header to get clean code (function reads the file itself)
                    clean_code = strip_header_from_code(code_path)
                    file_contents[model_name] = clean_code  # Key is just model name for clarity
                    self.logger.debug("Loaded code for model %s: %s", model_name, code_file)
                else:
                    self.logger.warning("Code file not found for model %s: %s", model_name, code_path)

        # Build directory listing
        from tools.helpers import _build_directory_listing
        dir_listing = _build_directory_listing(self.base_dir)

        # Load competition description
        description_path = self.base_dir / "description.md"
        if description_path.exists():
            with open(description_path, "r") as f:
                self.description = f.read()
        else:
            self.description = "No competition description available."

        return prompt_build_system(
            description=self.description,
            dir_listing=dir_listing,
            file_contents=file_contents,
            baseline_metadata=self.baseline_metadata,
            ensemble_strategy=self.strategy,  # Pass single strategy
            blacklisted_ideas=self.blacklisted_ideas,
        )

    def _call_sota_suggestions(self, **kwargs) -> str:
        """
        Override to pass is_ensemble=True for ensemble-specific SOTA suggestions.

        This ensures the suggestions follow ensemble-specific constraints:
        - Can change model family if beneficial
        - No changes to Data/Feature Engineering/Validation unless severe issues
        - Suggestions executable within 3 hours

        Args:
            **kwargs: Arguments to pass to search_sota_suggestions

        Returns:
            SOTA suggestions text with ensemble-specific prompts
        """
        from tools.developer import search_sota_suggestions

        # Add is_ensemble=True to kwargs
        kwargs['is_ensemble'] = True

        return search_sota_suggestions(**kwargs)

    def _get_code_timeout(self) -> int:
        """
        Override to use ensemble-specific code execution timeout.

        Returns:
            Timeout in seconds for code execution (10800 = 3 hours for ensemble)
        """
        from tools.developer import _ENSEMBLE_CODE_TIMEOUT
        return _ENSEMBLE_CODE_TIMEOUT
