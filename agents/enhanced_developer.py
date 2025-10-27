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
        super().__init__(
            slug=slug,
            iteration=iteration,
            model_name=model_name,
            model_recommendations=None,  # Not used for enhanced agent
            later_recommendations=None,
            cpu_core_range=cpu_core_range,
            gpu_identifier=gpu_identifier,
            gpu_isolation_mode=gpu_isolation_mode
        )

        # Override starting version to continue from baseline
        self._starting_version = self._get_starting_version()
        self.version = self._starting_version

        logger.info(
            "EnhancedDeveloperAgent initialized for %s (model=%s, starting_version=v%d)",
            slug, model_name, self._starting_version
        )
        logger.info("Base code: %s", self.base_code_path)
        logger.info("Enhancements: %s", self.enhancements_path)

    def _get_starting_version(self) -> int:
        """
        Auto-detect the latest version in the output directory and return next version.

        Example: If code_2_1_v3.py exists, return 4 (start from v4)

        Returns:
            Next version number (baseline_max + 1)
        """
        code_files = list(self.outputs_dir.glob("code_*_v*.py"))

        if not code_files:
            logger.warning("No existing code files found in %s, starting from v1", self.outputs_dir)
            return 1

        versions = []
        for f in code_files:
            match = re.search(r'_v(\d+)\.py$', f.name)
            if match:
                versions.append(int(match.group(1)))

        if not versions:
            logger.warning("No version numbers found in filenames, starting from v1")
            return 1

        next_version = max(versions) + 1
        logger.info("Detected baseline versions: %s, starting enhanced from v%d", versions, next_version)
        return next_version

    def _compose_system_prompt(self) -> str:
        """
        Override system prompt for first iteration to use enhancement-focused prompt.

        For subsequent iterations, falls back to standard DeveloperAgent prompt
        (which uses Red Flags + SOTA for refinement).

        Returns:
            System prompt string
        """
        # First iteration: Apply enhancements from markdown
        if self.version == self._starting_version:
            return self._compose_enhancement_system_prompt()

        # Subsequent iterations: Use inherited prompt (Red Flags + SOTA)
        # But we need to customize it to remove the single-fold constraint
        return self._compose_relaxed_system_prompt()

    def _compose_enhancement_system_prompt(self) -> str:
        """
        Generate enhancement-focused system prompt for first iteration.

        Reads base code and enhancements, instructs LLM to apply them.

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

    def _compose_relaxed_system_prompt(self) -> str:
        """
        Generate system prompt for subsequent iterations (v2+).

        Similar to DeveloperAgent but with relaxed constraints
        (allows multi-fold, ensembling, etc.)

        Returns:
            System prompt for refinement iterations
        """
        from prompts.developer_agent import build_system
        from tools.helpers import _build_directory_listing

        # Get directory listing
        directory_listing = _build_directory_listing(str(self.base_dir))

        # Read description
        description_path = self.base_dir / "description.md"
        with open(description_path, 'r') as f:
            description = f.read()

        # Build prompt with multi-fold allowed
        relaxed_prompt = build_system(
            description=description,
            directory_listing=directory_listing,
            model_name=self.model_name,
            model_recommendations="(Enhanced model - no specific recommendations, use Red Flags + SOTA for refinement)",
            slug=self.slug,
            cpu_core_range=self.cpu_core_range,
            gpu_identifier=self.gpu_identifier,
            gpu_isolation_mode=self.gpu_isolation_mode,
            allow_multi_fold=True
        )

        # Add note about enhanced context
        relaxed_prompt += """

**Enhanced Model Context:**
This is an enhanced version of a baseline model. Previous iterations have applied cross-model improvements.
Continue refining based on Red Flags analysis and SOTA suggestions. Multi-fold training and advanced
techniques (ensembling, stacking, calibration) are allowed if they improve the competition metric.
"""

        return relaxed_prompt
