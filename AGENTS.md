# Repository Guidelines

## Project Structure & Module Organization

P2PNet is a compact PyTorch crowd-counting project. `train.py` is the training entry point, `run_test.py` runs visualization inference, and `engine.py` contains the training and evaluation loops. Model definitions live in `models/`; dataset loading and augmentation live in `crowd_datasets/`, with ShanghaiTech Part A support under `crowd_datasets/SHHA/`. Shared distributed-training and tensor utilities are in `util/`. Use `tools/` for dataset conversion and training-monitor scripts. Demo images and architecture graphics belong in `vis/`; the tracked reference checkpoint is `weights/SHTechA.pth`.

There is currently no dedicated `tests/` directory. Keep generated logs, run data, caches, and experimental checkpoints out of source control as defined in `.gitignore`.

## Build, Test, and Development Commands

Create an isolated Python environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a syntax check with `python -m compileall -q crowd_datasets models tools util *.py`.

Train on a prepared SHHA-style dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --data_root /path/to/data \
  --dataset_file SHHA --output_dir logs --checkpoints_dir weights
```

Run the bundled demo with `python run_test.py --weight_path weights/SHTechA.pth --output_dir logs`. Dataset preparation helpers expose their options through, for example, `python tools/build_p2pnet_data.py --help`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants. Keep imports grouped as standard library, third-party, then local modules. Add concise docstrings to scripts and non-obvious public helpers. No formatter or linter is enforced; keep changes PEP 8-compatible and avoid unrelated reformatting.

## Testing Guidelines

No automated framework or coverage threshold is configured. At minimum, run the compile check and the relevant command with `--help`. For dataset changes, test both empty and non-empty annotations. For model or training changes, report the dataset, checkpoint, command, and before/after MAE and MSE. Place future pytest files under `tests/` using `test_*.py` names.

## Commit & Pull Request Guidelines

History uses short, imperative, sentence-style subjects such as `Improve training compatibility and tooling`. Keep each commit focused. Pull requests should explain the motivation and affected modules, list validation commands, link related issues, and include metric tables or visualization screenshots when behavior changes. Never commit private datasets, absolute local paths, credentials, TensorBoard logs, or generated checkpoints.
