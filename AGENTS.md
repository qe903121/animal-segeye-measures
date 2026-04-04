# AGENTS.md

Last updated: 2026-04-04

## Project Summary

This repository implements a reproducible animal-image metrology baseline on COCO.

Canonical operator flow:

- `data`
- `annotate`
- `review`
- `predict`
- `validate`

Canonical entry point:

```bash
python main.py [--config config/config.yaml] [--verbose] <command> [args]
```

## Read Order

1. [docs/01_architecture.md](./docs/01_architecture.md)
2. [docs/02_active_context.md](./docs/02_active_context.md)
3. [docs/03_dev_journal.md](./docs/03_dev_journal.md)
4. [system_architecture.md](./system_architecture.md)

## Document Ownership

- `README.md`: user-facing setup, workflow, and output artifacts
- `docs/01_architecture.md`: stable technical concepts, contracts, formulas
- `docs/02_active_context.md`: current status, TODOs, roadmap, acceptance
- `docs/03_dev_journal.md`: historical notes, migrations, resolved pitfalls
- `system_architecture.md`: diagram-centric system summary

## Maintenance Rule

Keep this file short.

- Do not duplicate workflow walkthroughs here.
- Do not duplicate formulas or methodology here.
- Update the owner document instead of expanding this file.
