DATA NOTICE

The file `data/downloaded/meddialog.json` is a large dataset (MedDialog) and was removed from this repository to keep the project lightweight and within GitHub limits.

To obtain the dataset:

- If you have permission to redistribute, place `meddialog.json` at `data/downloaded/meddialog.json`.
- Otherwise, download the canonical MedDialog dataset from the original source and save it at `data/downloaded/meddialog.json`.

Usage:
- The code expects `data/downloaded/meddialog.json` to exist for local runs and experiments.
- If you cannot store the dataset, set `max_samples` when running `run_experiment.py` to a smaller number or update `ui/app.py` to use a local sample.
