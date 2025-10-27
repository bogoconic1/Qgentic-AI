"""
EnhancedDeveloperAgent: Extends DeveloperAgent to apply cross-model enhancements.

This agent takes an existing baseline model implementation and enhances it by:
1. Applying "Add to pipeline" recommendations from cross-model analysis
2. Removing "Remove from pipeline" risky strategies
3. Continuing to use Red Flags + SOTA for iterative refinement

Key differences from DeveloperAgent:
- Starts from existing code (not from scratch)
- Relaxed constraints (allows multi-fold training, ensembling)
- Auto-detects starting version (continues from baseline's last version)
"""

import logging
import re
from pathlib import Path
from typing import Optional

from agents.developer import DeveloperAgent
from tools.helpers import call_llm_with_retry

logger = logging.getLogger(__name__)


class EnhancedDeveloperAgent(DeveloperAgent):
    """
    Specializes DeveloperAgent to apply cross-model knowledge transfer.

    Inherits full DeveloperAgent functionality, overrides only:
    - __init__: Accept enhancement inputs
    - _compose_system_prompt: Enhancement-focused prompt for first iteration
    - _get_starting_version: Auto-detect latest baseline version
    """

    def __init__(
        self,
        slug: str,
        iteration: int,
        model_name: str,
        base_code_path: str,
        enhancements_path: str,
        cpu_core_range: Optional[list[int]] = None,
        gpu_identifier: Optional[str] = None,
        gpu_isolation_mode: str = "none",
        **kwargs
    ):
        """
        Initialize enhanced developer agent.

        Args:
            slug: Competition slug
            iteration: Iteration number (e.g., "2_1" for model 1)
            model_name: Model name (e.g., "deberta-v3-large")
            base_code_path: Path to baseline code file (e.g., code_2_1_v3.py)
            enhancements_path: Path to enhancements markdown (e.g., enhancements_model_1.md)
            cpu_core_range: CPU cores to pin to
            gpu_identifier: GPU ID or MIG UUID
            gpu_isolation_mode: GPU isolation mode ("mig", "multi-gpu", "none")
        """
        # Store enhancement-specific paths before calling super().__init__
        self.base_code_path = Path(base_code_path)
        self.enhancements_path = Path(enhancements_path)

        if not self.base_code_path.exists():
            raise FileNotFoundError(f"Base code not found: {base_code_path}")
        if not self.enhancements_path.exists():
            raise FileNotFoundError(f"Enhancements not found: {enhancements_path}")

        # Initialize parent WITHOUT model_recommendations (we use enhancements instead)
        # Use iteration_enhanced for separate directory
        iteration_enhanced = f"{iteration}_enhanced"
        super().__init__(
            slug=slug,
            iteration=iteration_enhanced,
            model_name=model_name,
            model_recommendations=None,  # Not used for enhanced agent
            later_recommendations=None,
            cpu_core_range=cpu_core_range,
            gpu_identifier=gpu_identifier,
            gpu_isolation_mode=gpu_isolation_mode
        )

        # Store original iteration for reference
        self.original_iteration = iteration

        # Override starting version to start from v1 (fresh start in enhanced directory)
        self._starting_version = 1
        self.version = self._starting_version

        logger.info(
            "EnhancedDeveloperAgent initialized for %s (model=%s, iteration=%s, starting_version=v%d)",
            slug, model_name, iteration_enhanced, self._starting_version
        )
        logger.info("Base code: %s", self.base_code_path)
        logger.info("Enhancements: %s", self.enhancements_path)

    def _compose_system(self) -> str:
        """
        Generate enhancement-focused system prompt.

        Reads base code and enhancements, instructs LLM to apply them.
        Called once at the start - subsequent iterations use Red Flags + SOTA loop.

        Returns:
            System prompt with enhancements
        """
        from prompts.developer_agent import build_enhancement_system

        # Load base code
        with open(self.base_code_path, 'r') as f:
            base_code = f.read()

        # Load enhancements
        with open(self.enhancements_path, 'r') as f:
            enhancements = f.read()

        # Read competition description
        description_path = self.base_dir / "description.md"
        with open(description_path, 'r') as f:
            description = f.read()

        return build_enhancement_system(
            description=description,
            model_name=self.model_name,
            base_code=base_code,
            enhancements=enhancements,
            outputs_dir=str(self.outputs_dir),
            cpu_core_range=self.cpu_core_range
        )

    def _call_sota_suggestions(self, **kwargs):
        """
        Override to pass allow_multi_fold=True to SOTA suggestions tool.

        This allows SOTA suggestions in subsequent iterations (v2, v3, ...)
        to include multi-fold training, ensembling, stacking, and calibration.

        Args:
            **kwargs: Arguments to pass to search_sota_suggestions

        Returns:
            SOTA suggestions text
        """
        from tools.developer import search_sota_suggestions

        # Add allow_multi_fold=True to the kwargs
        kwargs['allow_multi_fold'] = True
        return search_sota_suggestions(**kwargs)
