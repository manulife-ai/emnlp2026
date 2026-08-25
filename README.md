# Harbor Trial Test Data Usage

This guide documents `data/trials_data.parquet`, the released Harbor trial
cache used for task-testing analysis and optional cache-backed evaluation. It
is historical execution data, not a standalone task runner: it contains
results from earlier Harbor run trajectories and meant for research comunity to train new skill routers or perform skill retrieval researches. It does not contain the full benchmark environments, or solver trajectories themselves.

## Data Release Statistics

The trial data file statistics are as follows.

| Check | Result |
|---|---:|
| Rows / columns | 1,423 / 29 |
| Benchmarks | 609 SkillsBench rows, 814 Terminal-Bench rows |
| Unique task names | 173 |
| Unique `(task_name, trial_name)` keys | 1,423 |
| Unique `trial_uri` values | 1,423 |
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

The harvest code records two related but non-equivalent quantities:

- **`reward`:** the Harbor result scalar for the trial. It is the value used
  by `HarborCache` when replaying a cached result. It should be interpreted
  together with `exception_type`, because a reward recorded on an exception
  row may not represent a completed verifier run.
- **`soft_reward`:** a post-processed test-level score. For a CTRF file with
  `n_tests_total` tests, it is

  $$\mathrm{soft\_reward} = \frac{\mathrm{n}_{\mathrm{tests\_passed}}}{\mathrm{n}_{\mathrm{tests\_total}}}.$$

  Skipped tests are excluded from `n_tests_failed` but remain part of
  `n_tests_total`, matching the harvest implementation. If CTRF is missing or
  contains no tests, `soft_reward` and the test-count fields are null.

For retrieval-data construction, the training scripts define:

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

## Reproducibility Advice
To facilitate future researcher's work, please consider recording the input checksum, script versions, seed, positive threshold, split fractions, Terminal-Bench policy, exception policy, and whether the model used
`mean_effective_reward` or `label`. The generated `task_split.json` should be
published with any derived training data so task-level isolation can be
audited.