# LanceDB MCAP robotics data curation demo

An end-to-end notebook showing how to turn public nuImages driving data into MCAP logs, ingest them into LanceDB, curate training rows with LanceDB Feature Engineering, and fine-tune a small vision-language model for driving-scene tagging.

The experiment compares the vanilla model with full-model training on all frames and on a quality-filtered subset. The checked-in notebook includes the populated reference results.

## Run it

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab nbformat
jupyter lab robotics_data_curation_post_training.ipynb
```

Run the notebook from top to bottom. It installs any remaining Python packages into the active environment and keeps downloads, generated data, model caches, and results inside the project directory.

The first run downloads roughly 118 MB of nuImages data, a 45 MB feature extractor, and a 0.5 GB vision-language model. Apple Silicon or an NVIDIA GPU is recommended for training. CPU preprocessing works normally; to explicitly enable the slower CPU training path, start Jupyter with `ALLOW_CPU_FULL_TRAINING=1`.

## Files

- `robotics_data_curation_post_training.ipynb` — the runnable experiment and reference output.
- `build_notebook.py` — regenerates the notebook source with `python build_notebook.py`.

This is a compact research demonstration of scene tagging, not a vehicle-control, detection, or sensor-fusion system.
