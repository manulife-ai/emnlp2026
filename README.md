# When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents
This Readme file consists of preparation guides for the two datasets employed in our paper: **Harbor Trial dataset** which consists of real data and **Track B** dataset which contains synthetic data. You can follow this guide to finetune your own solutions and recipes using our paper's data.

# Harbor Trial Dataset Usage
This guide documents the usage of `data/trials_data.parquet`, the Harbor trials we used in our paper. It contains task execution tracjectories and results for research comunity to train new skill routers or perform skill retrieval researches. Note that you will need to install and configure the Harbor envinronment on your own compute from [https://www.harborframework.com](https://www.harborframework.com).

## Harbor Data Statistics

The trial data file statistics are as follows. There are in total **1084 valid trials** (which return valid outcomes without exception).

| Item | Result |
|---|---:|
| Total Rows / columns | 1,423 / 29 |
| **Valid Trials (without exception)** | **1084** | 
| Benchmarks | 609 SkillsBench rows, 814 Terminal-Bench rows |
| Unique task names | 173 |
| Missing task or trial identifiers | 0 |
| Missing rewards | 208 |
| Missing soft rewards and test counts | 307 |
| Rows with exceptions | 339 |
| Empty `injected_skills` lists | 562 |
| Missing `cost_usd` | 1,423 |

The file has no duplicate trial keys, invalid soft-reward ranges, invalid test
counts, or malformed JSON in the JSON-encoded columns. The following are
important semantic issues, not file-corruption errors:

1. `reward` and `soft_reward` are different signals. `reward` is the scalar
  Harbor recorded for the trial; it is extracted from
  `verifier_result.rewards.reward`, then top-level `reward`, then
  `verifier/reward.txt`. It is not guaranteed to be binary: 39 rows contain
  fractional values.
2. `soft_reward` is a derived verifier-coverage metric, computed as
  `n_tests_passed / n_tests_total` from `verifier/ctrf.json`. It is in
  `[0, 1]` and is available only when a usable CTRF test list exists.
3. `reward > 0` and `soft_reward > 0` are collected and calculated to represent different angle of the trial's outcome. 
   Do not derive one from the other. They represent separate recorded signals.
4. There are 339 exception rows. Some have a recorded reward, but those
   rewards may describe a partial or failed execution. Exclude exception rows
   for clean performance summaries unless the failure analysis is the goal.
5. Seventy-nine task names occur in both benchmarks. Always group or join by
   `(benchmark, task_name)`, never by `task_name` alone.
6. `cost_usd` is entirely null in this release. Token columns are available
   for most rows, but cost analysis cannot be reproduced from this file alone.
7. `trajectory_path` values are historical source paths and may not exist on
   another machine. The parquet does not ship the referenced trajectories.

## Schema

The Parquet schema is:

| Column | Meaning |
|---|---|
| `task_name` | Task identifier. Not globally unique across benchmarks. |
| `trial_name` | Harbor trial identifier, normally containing `__`. |
| `trial_uri` | Trial URI; unique in this file. |
| `task_checksum` | Checksum of the task definition. |
| `benchmark` | `skillsbench` or `terminalbench`. |
| `agent_name` | Agent implementation name. |
| `model_name` | Solver model name. |
| `env_type` | Execution environment, such as `docker` or `daytona`. |
| `reward` | Harbor-recorded scalar, loaded from `verifier_result.rewards.reward`, top-level `reward`, or `verifier/reward.txt`; may be null, binary, or fractional. |
| `soft_reward` | Derived verifier coverage: `n_tests_passed / n_tests_total`; null when CTRF test results are unavailable or empty. |
| `n_tests_total` | Number of verifier tests observed. |
| `n_tests_passed` | Number of passed verifier tests. |
| `n_tests_failed` | Number of failed verifier tests, excluding skipped tests. |
| `failed_test_names` | Native Parquet list of failed or non-passed test names. |
| `exception_type` | Harbor or environment exception type, when present. |
| `exception_message` | Truncated exception text. |
| `started_at`, `finished_at` | ISO-8601 execution timestamps stored as strings. |
| `n_input_tokens`, `n_output_tokens`, `n_cache_tokens` | Token counts. |
| `cost_usd` | Cost field; null for every released row. |
| `trajectory_path` | Original trajectory path, not a portable bundled file. |
| `n_steps` | Number of trajectory steps. |
| `tool_call_counts` | JSON string containing a tool-name-to-count object. |
| `n_skill_invocations` | Number of `Skill` tool calls. |
| `skill_invocation_args` | JSON string containing a list of invocation arguments. |
| `injected_skills` | JSON string containing the skills placed in the task sandbox. |
| `source_root` | Original harvest source root; generally not portable. |

`failed_test_names` is a native list column. The other structured columns
listed as JSON strings must be decoded with `json.loads` before use.

## Reward Columns

Our harbor dataset records two related but non-equivalent quantities:

- **`reward`:** the raw Harbor result scalar for the trial. It is the value used
  by `HarborCache` when replaying a cached result. It should be interpreted
  together with `exception_type`, since a row with exception indicates the task isn't completed successfully.
  
- **`soft_reward`:** a post-processed result score. For a CTRF file with
  `n_tests_total` tests, it is

  $$\mathrm{soft\_reward} = \frac{\mathrm{n}_{\mathrm{tests\_passed}}}{\mathrm{n}_{\mathrm{tests\_total}}}.$$

  Skipped tests are excluded from `n_tests_failed` but remain part of
  `n_tests_total`, matching the harvest implementation. If CTRF is missing or
  contains no tests, `soft_reward` and the test-count fields are null.

For training data construction, our protocol define:

```python
effective_reward = soft_reward.fillna(reward)
```

Thus `soft_reward` is preferred as the task-skill supervision value and is
also used for positive thresholds such as `mean_soft_reward > 0.5`; it is not
merely a display or filtering field. The raw `reward` is the fallback when
`soft_reward` is unavailable. Rows where both values are missing are omitted
from task-skill training pairs. This `effective_reward` rule is separate from
the Harbor cache, which loads the raw `reward` field.

## Standalone Release Contents

Researchers need only these files from this release:

- `data/trials_data.parquet`
- `preprocess_trials.py`
- `build_training_pairs.py`

The scripts use only Python, pandas, and PyArrow. They do not import the
private skill-router package, Harbor, benchmark task directories, search
indexes, model checkpoints, or repository-specific configuration.

## Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow
```

## Preprocess the Trial Data

Run this first. It validates required columns and duplicate trial identities,
decodes JSON columns, adds `effective_reward`, and marks clean/scored and
positive-candidate rows.

```bash
python preprocess_trials.py \
  data/trials_data.parquet \
  output/normalized_trials.parquet
```

Important generated columns:

- `effective_reward`: `soft_reward` when present, otherwise `reward`.
- `is_clean_scored`: no exception and a non-null effective reward.
- `is_positive_candidate`: clean scored row with effective reward above 0.5.
- `injected_skills_list`, `tool_call_counts_map`, and
  `skill_invocation_args_list`: decoded structured values.
- `benchmark_task_id`: collision-safe benchmark/task identifier.

## Build Training Pairs and Splits

```bash
python build_training_pairs.py \
  output/normalized_trials.parquet \
  output/training_pairs
```

The script removes exception rows and rows without injected skills, explodes
each injected skill into a task-skill observation, and averages repeated
observations by `(benchmark, task_name, skill_id)`. It writes:

- `all_pairs.parquet`: every aggregated task-skill pair.
- `train_pairs.parquet`: training partition.
- `validation_pairs.parquet`: validation partition.
- `test_pairs.parquet`: test partition.
- `task_split.json`: reproducible task-level split metadata.

The default split is 70% train, 15% validation, and 15% test for each
benchmark. All rows for a benchmark-qualified task remain in one partition.
Terminal-Bench tasks are test-only by default, preserving the OOD evaluation
policy. To opt into splitting Terminal-Bench across train, validation, and
test, pass `--include-terminalbench-train`.

The pair table contains:

- `mean_effective_reward`: the training relevance signal after fallback.
- `n_trials`: number of clean trial observations for the pair.
- `is_positive`: `mean_effective_reward > 0.5` by default.
- `label`: integer form of `is_positive`, suitable for a binary router loss.
- `split`: `train`, `validation`, or `test`.

Change the threshold or split seed explicitly when running an experiment:

```bash
python build_training_pairs.py \
  output/normalized_trials.parquet \
  output/training_pairs \
  --positive-threshold 0.5 \
  --seed 42
```

## Use the Outputs in a New Skill Router

Load `train_pairs.parquet` for model fitting and use
`validation_pairs.parquet` for threshold or hyperparameter selection. Keep
`test_pairs.parquet` untouched until final reporting. Use `task_id` or the
`(benchmark, task_name)` columns as the grouping key; do not group by
`task_name` alone.

For soft-label training, use `mean_effective_reward`. For binary training,
use `label`. Keep `n_trials` as a confidence/count feature if desired. Do not
use `reward` and `soft_reward` interchangeably, and do not treat exception
rows as ordinary negative examples without a deliberate failure model.

This release provides historical trial outcomes and preprocessing targets. It
does not provide task instructions, skill documents, a retrieval index, a
Harbor runner, or a cache adapter. A researcher building a new router must
supply their own task and skill representations, retrieval model, training
loop, and evaluation runner.


# Track B Dataset Usage

This directory is the Track B data-only release for the continual-learning-style skill retrieval experiments. It contains the locked training, validation, and evaluation data used in the Ring 1, Ring 2 and Ring 3 runs mentioned in the paper.

The purpose of releasing the data is to give researchers opportunities to design their own skill retriever and rerankers and explore their new mitigation methodologies for catastrophic forgetting phenomenon, and evaluate on the common benchmark to ground the performance of their work.

## Contents

| File | Rows | Unique Task Queries | Purpose |
|---|---:|---:|---|
| `train.parquet` | **13,271** | 5,874 |Track B training split |
| `val.parquet` | 556 | 149 | Validation split and checkpoint-selection data |
| `eval_set.parquet` | 78 | **21** | Ring 1 real in-distribution evaluation set |
| `synthetic_eval_set.parquet` | 4,023 | **2,414** | Ring 2 held-out-skill synthetic evaluation set |
| `ood_eval_set.parquet` | 15 | **10** | Ring 3 Terminal-Bench 2 OOD evaluation set |
| `positives.parquet` | 255 | 68 | Tiered real positive task-skill pool |
| `negatives.parquet` | 1,239 | 47 | Hard-negative pool |
| `track_b_quality_gates.json` | - | - | Aggregate quality-gate audit |
| `track_b_second_judge_audit.json` | - | - | Aggregate second-judge audit |

The two JSON files are supplementary audit metadata. They are not required by the training or evaluation scripts. They are included to document the dataset checks and known limitations, including reward-distribution drift, sparse skill coverage, and modest inter-judge agreement.

## Reproduce Track B setting with your own finetuning approach

The commands below are run from the root of the source repository, with this directory passed as the dataset directory. If this folder is downloaded separately, set `TRACK_B_DATA` to its local path.

```bash
export TRACK_B_DATA="$PWD/data/trackB"
```

### Prerequisites

1. Use Python 3.12 and install the repository dependencies from `requirements.txt` or `requirements-linux.txt`.
2. Download the external 34,396-skill catalog and retrieval index using the repository's `skill_router/scripts/download_search_index.py` script.
3. Ensure the base model `Qwen/Qwen3-Embedding-0.6B` is available. The evaluation scripts also use the frozen embedding baselines and the full skill catalog.

The external skill catalog and index are intentionally not duplicated in this data release. They are required to evaluate retrieval against the full skill pool.

### Finetuning and Evaluation on synthetic test set

You can finetune using your methodology with the following settings described in our paper:


## Evaluation rings

- **Ring 1:** `eval_set.parquet`, real in-distribution SkillsBench evaluation.
- **Ring 2:** `synthetic_eval_set.parquet`, held-out-skill synthetic evaluation.
- **Ring 3:** `ood_eval_set.parquet`, Terminal-Bench 2.0 out-of-distribution evaluation.

One example of finetuning and evaluating on Terminal-Bench 2.0 data with OOD distribution is as follows: 

```bash
python your_finetune_methodology.py \
	--train-path "$TRACK_B_DATA/train.parquet" \
	--val-path "$TRACK_B_DATA/val.parquet" \
	--ckpt-dir ./output/smoke_ckpt \
	--eval-path "$TRACK_B_DATA/ood_eval_set.parquet" 
```

## Load and preprocess the data

The release includes two small utilities that require only Python, pandas, and a parquet engine such as `pyarrow`:

- `load_trackb.py` loads individual parquet files and validates the complete release against the expected row counts and model-facing columns.
- `preprocess_trackb.py` validates the release and converts selected parquet splits to newline-delimited JSON (`.jsonl`). Training rows retain their nested `negatives` records; evaluation rows retain their evaluation columns.

Run the release validation:

```bash
python data/trackB/load_trackb.py \
	--data-dir data/trackBdata
```

Print the same report as machine-readable JSON:

```bash
python data/trackB/load_trackb.py \
	--data-dir data/trackBdata \
	--json > trackb_validation.json
```

Convert all training and evaluation splits to a separate `processed/` directory:

```bash
python data/trackB/preprocess_trackb.py \
	--data-dir data/trackBdata \
	--output-dir ./trackb_processed
```

Convert only selected splits:

```bash
python data/trackB/preprocess_trackb.py \
	--data-dir data/trackBdata \
	--output-dir ./trackb_processed \
	--split train.parquet \
	--split eval_set.parquet
```

Use `--validate-only` when you want the schema and row-count checks without producing JSONL files. The original parquet files are never modified.

For Python users, import the loader directly:

```python
from data.trackB.load_trackb import load_split, training_records

train = load_split("train.parquet", "data/trackB")
records = training_records("data/trackB")
print(train.shape, records[0].keys())
```

## Reproducibility Advice
To facilitate future researcher's work, please consider recording the input checksum, script versions, seed, positive threshold, split fractions.

## Credits and Citations

If you use this dataset or the accompanying resources, please cite our paper
in the proceedings of EMNLP:

```bibtex
@inproceedings{murtaza2026synthetic,
  title     = {When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents},
  author    = {Syed Shariyar Murtaza, Yifan Nie, Utkarsh Soni, Eugene Wen and Arvid Frydenlund},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```
