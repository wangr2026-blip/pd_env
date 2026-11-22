"""Config-driven interpretability pipeline for frozen voice encoders.

Each stage is a CLI module: ``python -m pipeline.<stage> --config <yaml>``.
Stage outputs are cached under ``<output_root>/<run_name>/<stage>/<key>/``,
where the key hashes the stage's config sections, upstream keys and code.
The sealed final-test split is never loaded by any stage.
"""

__version__ = "0.1.0"
