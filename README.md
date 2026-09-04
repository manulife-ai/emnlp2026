# When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents
This Readme file serves as a guide for using the two datasets we presented in our paper: **Harbor Trial dataset** which consists of real task execution data and **Track B** dataset which contains synthetic data. Those datasets are intended to finetune your own skill retrieval solutions and recipes and serve as a common ground to compare new catastrophic forgetting mitigation methodologies with our proposed methods.

This Readme also has an example lora finetuning code to mimic the conservative anchor regularization setting described in the Table 3 of our paper.

# Dataset 1: Harbor Trial Dataset
This section documents the usage of `data/trials_data.parquet`, the Harbor trials we used in our paper. It contains task execution tracjectories and results we collected for research comunity to train new skill routers or perform skill retrieval researches.

## Harbor Data Statistics

The trial data file statistics are as follows. There are in total **1084 valid trials** (which return valid outcomes without exception).

| Item | Result |
|---|---:|
| Total Rows / columns | 1,423 / 29 |
| **Total Rows with Soft Rewards** | **1,116** | 
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

1. `soft_reward` is a derived metric, computed as
  `n_tests_passed / n_tests_total` from `verifier/ctrf.json`. It is in
  `[0, 1]` and is available only when a usable CTRF test list exists.
2. `soft_reward` is the preferred outcome signal to construct the finetuning data, always use `soft_reward` than `reward` unless `soft_reward` is not available for certain rows. 
   In that case, use `reward` as a surrogate.
3. There are 339 exception rows. Some have a recorded reward, but those
   rewards may describe a partial or failed execution. Exclude exception rows
   for clean performance summaries unless the failure analysis is the goal.
4. 79 task names occur in both benchmarks. Always group or join by
   `(benchmark, task_name)`, never by `task_name` alone.

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

- **`soft_reward`:** a post-processed result score. For a CTRF file (Harbor's Common Test Report file depicting the outcome of a task execution) with
  `n_tests_total` tests, it is

  $$\mathrm{soft\_reward} = \frac{\mathrm{n}_{\mathrm{tests\_passed}}}{\mathrm{n}_{\mathrm{tests\_total}}}.$$

  Skipped tests are excluded from `n_tests_failed` but remain part of
  `n_tests_total`, matching the harvest implementation. If CTRF is missing or
  contains no tests, `soft_reward` and the test-count fields are null.

For training data construction, `soft_reward` is preferred as the task-skill supervision signal. The raw `reward` is the fallback when
`soft_reward` is unavailable. Rows where both values are missing are omitted from task-skill training pairs.

## Harbor Data Preparation Scripts

- `data/trials_data.parquet`
- `preprocess_trials.py`
- `build_training_pairs.py`

This set of scripts prepares the harbor trials dataset in the same format that our papers uses. Please run in the following steps.

## Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow
```

## Preprocess the Trial Data

This step validates required columns decodes JSON columns, adds `effective_reward`, and marks clean/scored and
positive-candidate rows.

```bash
python preprocess_trials.py \
  data/trials_data.parquet \
  output/normalized_trials.parquet
```

The key columns added after the processing:

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

This step removes exception rows and rows without injected skills, explodes
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
nature. To opt into splitting Terminal-Bench across train, validation, and
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


# Dataset 2: Track B Dataset used in our paper

This section of the guide describes the Track B dataset used in our paper. It contains the training, validation, and evaluation data used in the Ring 1 (in-distribution real test data), Ring 2 (in-distribution synthetic test data) and Ring 3 (OOD real test data) runs described in the paper.

The purpose of releasing this set data is to give researchers opportunities to design their own skill retriever and rerankers and explore their new mitigation methodologies for catastrophic forgetting phenomenon, and evaluate on the common benchmark to ground the performance of their work.

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

## Run Track B setting with your own finetuning approach

To finetune and test your own skill retrieval/reranking system using our data, follow the below steps. 
First, set `TRACK_B_DATA` to its local path for easy path references in future commands.

```bash
export TRACK_B_DATA="./data/trackB"
```

### Python libs and environment setups

1. Use Python 3.12 and install the repository dependencies from `requirements.txt` or `requirements-linux.txt`.
2. Ensure the base model `Qwen/Qwen3-Embedding-0.6B` [Link](https://huggingface.co/Qwen/Qwen3-0.6B) is available, since this is the baseline model to be compared with in our paper. It is recommended to be compared against your new skill retrival/reranking system as well.


### Finetuning and Evaluation on synthetic test set

You can finetune using your methodology with the following settings described in our paper:

## Training Data for finetuning
`data/trackB/train.parquet`

## validation data
`data/trackB/val.parquet`

## Evaluation Data for Different Rings (Scenarios)

- **Ring 1:** `eval_set.parquet`, real in-distribution SkillsBench evaluation.
- **Ring 2:** `synthetic_eval_set.parquet`, held-out-skill synthetic evaluation.
- **Ring 3:** `ood_eval_set.parquet`, Terminal-Bench 2.0 out-of-distribution evaluation.

## Load and preprocess the data

The scripts contains preprocessing codes to format the the data in the same way that our papaer used.

- `load_trackb.py` loads individual parquet files and validates the dataframe against the expected row counts and model-facing columns, to check data integrity.
- `preprocess_trackb.py` validates the release and converts selected parquet splits to newline-delimited JSON (`.jsonl`). Training rows retain their nested `negatives` records; evaluation rows retain their evaluation columns.

Run the following data processing commands:

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

Convert only selected splits (e.g. training fold on):

```bash
python data/trackB/preprocess_trackb.py \
	--data-dir data/trackBdata \
	--output-dir ./trackb_processed \
	--split train.parquet
```

Once the data preprocessing is done, you can use the data in the output jsonl file to finetune your own skill retriever/reranker and evaluate its performance on this common dataset that we presented in our paper.

# Sample Script for Conservative-Anchor-based Finetuning

`trackb_cons_anchor_example.py` reproduces the **conservative embedding-anchor** rows of Table 3 from *When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents*.

It is intentionally limited to one forgetting-mitigation method: attention-only LoRA fine-tuning with an embedding-anchor loss (Cons. Anchor approach listed in Table 3 of our paper). It can run any combination of the paper's Track B data configurations:

| Config | Training data |
| --- | --- |
| `A` | Full mix: real positives, paraphrases, and skill-first synthetic rows |
| `B` | Full mix except skill-first synthetic rows |
| `C` | Skill-first synthetic rows only; 10% of its query IDs are held out for validation |

## Requirements

Run commands from the repository root (`skill_router`). The script requires:

- Python with `torch`, `transformers`, `peft`, `numpy`, `pandas`, and `pyarrow`.
- CUDA is recommended for model fine-tuning.
- The released Track B files in `data/trackBdata`.
- The full skill retrieval corpus, which is not part of the parquet release:
  - `skillusage/skills/skills_meta.jsonl`
  - `skillusage/search_server/index/skills.db`
- Access to the Hugging Face model `Qwen/Qwen3-Embedding-0.6B`.

First create your own virtual environment and install the dependencies described in requirements.txt
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Download the required metadata and pre-built retrieval index via the helper script:

```bash
python scripts/release/download_trackb_index.py
```

The downloader fetches only `skills_meta.jsonl` and the prebuilt index from the
`Shiyu-Lab/Skill-Usage` Hugging Face repo. Please make sure you have active internet connection to download from Hugging Face.

## Validate The Release

Before training, verify that the expected parquet files, schemas, and locked row counts are present:

```bash
python scripts/release/trackb_cons_anchor_example.py --validate-only
```

A successful run ends with:

```text
release_valid = True
```

## Run Conservative-Anchor Regularized Finetuning


For example, run a single configuration (Config A):

```bash
python scripts/release/trackb_cons_anchor_example.py \
  --data-dir data/trackBdata \
  --out-dir output/trackb_cons_anchor \
  --configs A
```

Run selected configurations only:

```bash
python scripts/release/trackb_cons_anchor_example.py \
  --data-dir data/trackBdata \
  --out-dir output/trackb_cons_anchor \
  --configs A,C
```

The default recipe is the Table 3 conservative setting: LoRA rank 8, alpha 16, attention projections only, learning rate `5e-6`, one epoch, and anchor regularization weight `0.1`.

## Outputs And Resuming

The output directory contains:

- `adapters/<config>/anchor/`: LoRA checkpoints, checkpoint log, completion marker, and selected checkpoint.
- `results.json`: per-query evaluation metrics for each completed configuration.
- `selections.json`: selected checkpoint metadata.
- `table3_comparison.json`: reproduced Recall@10 values, paper values, deltas, and tolerance status.

The script skips a configuration that already has results for all three evaluation rings. Training checkpoints are also retained, so rerunning the same command reuses completed work.

Print the current result table without training:

```bash
python scripts/release/trackb_cons_anchor_example.py \
  --out-dir output/trackb_cons_anchor \
  --report-only
```

## Evaluation

Each selected adapter is evaluated over the full retrieval corpus on three rings using Recall@10:

| Ring | File | Evaluation set |
| --- | --- | --- |
| Ring 1 | `eval_set.parquet` | SkillsBench real in-distribution tasks |
| Ring 2 | `synthetic_eval_set.parquet` | Held-out-skill synthetic tasks |
| Ring 3 | `ood_eval_set.parquet` | Terminal-Bench 2 real OOD tasks |


Small variations are expected because Ring 1 and Ring 3 contain only 21 and 10 queries, respectively.

## Useful Options

```text
--lambda FLOAT             Anchor-loss coefficient; default: 0.1
--epochs INT               Fine-tuning epochs; default: 1
--save-every INT           Optimizer steps between checkpoints; default: 100
--lora-r INT               Override LoRA rank; default: 8
--lora-alpha INT           Override LoRA alpha; default: 16
--lr FLOAT                 Override learning rate; default: 5e-6
--seed INT                 Random seed; default: 42
--max-eval-queries INT     Evaluation cap for smoke tests only
```

Use `--max-eval-queries` only for a smoke test. It changes the reported metrics and should not be used for a Table 3 comparison.


# Credits and Citations

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
