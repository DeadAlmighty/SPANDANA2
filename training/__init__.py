"""SPANDANA training package — dataset preparation, models, evaluation.

Every script in this package is runnable directly from the VS Code terminal:

    python training/extract_lvh.py
    python training/train_cnn.py --epochs 40

Shared configuration lives in the single file ``config.json`` at the project
root (see ``training/config.py``) — no dataset or artifact path is ever
hard-coded inside the individual scripts.
"""
