# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run training: `python training/train.py [model.name="model_name"] [additional overrides]`
- Run training w/ slurm: `bash scripts/run_slurm.sh [model_name] [additional overrides]`
- Resume from checkpoint: Add `trainer.ckpt_path=checkpoints/[model_name]/last.ckpt`
- Local training script: `bash scripts/train.sh [model_name] [additional overrides]`

## Code Style
- **Imports**: Built-ins first, then third-party, then local imports (alphabetized)
- **Formatting**: 4-space indentation, line length ~100 chars
- **Naming**: PascalCase for classes, snake_case for functions/variables
- **Typing**: No strict typing pattern established
- **Error handling**: Use assertions for validation, raise ValueError with context
- **Docstrings**: Write docstrings for main classes and methods
- **Logging**: Use the logging module with semantic emoji prefixes (e.g., 📁, 🚀, 💾)
- **Configuration**: Use Hydra for config management with YAML files

Always follow existing patterns in the codebase for consistency.