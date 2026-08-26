# Repository Guidelines

## Project Structure & Module Organization

`train.py` starts training, `run_test.py` runs visualization inference, and `engine.py` holds training and evaluation loops. Models live in `models/`; dataset loading and augmentation live in `crowd_datasets/`, including ShanghaiTech Part A under `crowd_datasets/SHHA/`. Shared utilities are in `util/`, while `tools/` contains dataset conversion and monitoring scripts. Keep demo assets in `vis/`; the tracked reference checkpoint is `weights/SHTechA.pth`.

There is no dedicated `tests/` directory. Keep generated logs, run data, caches, and experimental checkpoints out of source control as defined in `.gitignore`.

## Build, Test, and Development Commands

Create an environment and install dependencies:

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

Run the bundled demo with `python run_test.py --weight_path weights/SHTechA.pth --output_dir logs`. Inspect dataset helper options with `python tools/build_p2pnet_data.py --help`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants. Keep imports grouped as standard library, third-party, then local modules. Add concise docstrings to scripts and non-obvious public helpers. No formatter or linter is enforced; keep changes PEP 8-compatible and avoid unrelated reformatting.

## Testing Guidelines

No automated framework or coverage threshold is configured. Run the compile check and relevant command with `--help`. For dataset changes, test empty and non-empty annotations. For model or training changes, report the dataset, command, and before/after MAE and MSE. Put future pytest files under `tests/` as `test_*.py`.

## Experiment Bank Workflow

Use `experiment-bank/` as the experiment ledger. `task.md` lists work using only `pending` and `compeleted`; add new tasks as `pending` and mark them `compeleted` only after the run ends. Store each experiment's rationale in `ideas/` and its configuration, metrics, observations, and outcome in `experiments/`. Use matching descriptive filenames, such as `ideas/multistep-lr.md` and `experiments/multistep-lr.md`. After every experiment, update all applicable records, including failed or inconclusive runs.

## Commit & Pull Request Guidelines

History uses short, imperative, sentence-style subjects such as `Improve training compatibility and tooling`. Keep each commit focused. Pull requests should explain the motivation and affected modules, list validation commands, link related issues, and include metric tables or visualization screenshots when behavior changes. Never commit private datasets, absolute local paths, credentials, TensorBoard logs, or generated checkpoints.
