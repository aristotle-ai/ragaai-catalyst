# Deleted Modules History

This document tracks modules that have been removed from the RagaAI-Catalyst package. All deleted code is preserved in git history and can be recovered if needed.

## Deletion Record

**Deletion Date:** 2025-12-03  
**Last Commit Before Deletion:** `9e4452f9e0eb9649a0df08ef7b0faab088f1193d`

## Deleted Modules

### 1. synthetic_data_generation

**Path:** `ragaai_catalyst/synthetic_data_generation.py`

**Description:** Module for generating synthetic data for testing and training purposes.

**Size:** 832 lines

**Related Files:**
- Test file: `tests/test_catalyst/test_synthetic_data_generation.py`

---

### 2. redteaming

**Path:** `ragaai_catalyst/redteaming/`

**Description:** Red teaming module for security testing and adversarial evaluation of AI systems.

**Directory Structure:**
- `__init__.py` - Module initialization
- `config/` - Configuration files
- `data_generator/` - Data generation utilities
- `evaluator.py` - Evaluation logic (4599 bytes)
- `llm_generator.py` - LLM generation logic (5468 bytes)
- `llm_generator_old.py` - Legacy generator (3396 bytes)
- `red_teaming.py` - Main red teaming implementation (15081 bytes)
- `requirements.txt` - Module-specific dependencies
- `tests/` - Module tests
- `upload_result.py` - Result upload utilities (894 bytes)
- `utils/` - Utility functions

**Dependencies (from redteaming/requirements.txt):**
- openai>=1.0.0
- pandas>=2.0.0
- tomli>=2.0.0
- tqdm>=4.65.0

---

## Recovery Instructions

To recover any of these deleted modules from git history:

### View deleted file contents:
```bash
# For synthetic_data_generation.py
git show 9e4452f9e0eb9649a0df08ef7b0faab088f1193d:ragaai_catalyst/synthetic_data_generation.py

# For redteaming module files
git show 9e4452f9e0eb9649a0df08ef7b0faab088f1193d:ragaai_catalyst/redteaming/red_teaming.py
```

### Restore a deleted file:
```bash
# Restore synthetic_data_generation.py
git checkout 9e4452f9e0eb9649a0df08ef7b0faab088f1193d -- ragaai_catalyst/synthetic_data_generation.py

# Restore entire redteaming directory
git checkout 9e4452f9e0eb9649a0df08ef7b0faab088f1193d -- ragaai_catalyst/redteaming/
```

### View file history:
```bash
# See all commits that modified a deleted file
git log --all --full-history -- ragaai_catalyst/synthetic_data_generation.py
git log --all --full-history -- ragaai_catalyst/redteaming/
```

## Notes

- All dependencies specific to these modules (openai, pandas, tomli, tqdm) are still used by other parts of the package, so they remain in the main requirements.
- No other modules in the codebase had dependencies on these deleted modules.
- The deletion was clean with no broken imports remaining.
