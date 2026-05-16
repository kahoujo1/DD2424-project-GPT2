# Experiment Pipeline Guide

Use `run_experiments.py` to launch multiple training/evaluation experiments and store the results in one place.

The pipeline creates:

- `experiments/results.csv`: summary table of all runs. This file is not replaced automatically; new runs are appended to the CSV.
- `experiments/logs/`: full terminal output for each run.
- `experiments/plots/`: generated plots from completed runs.

Before running real experiments, use dry-run mode to check that everything works:

```bash
python3 run_experiments.py --experiment comparison --dry-run
```

Dry-run prints the commands without training anything.

Training pipeline is basically this:

run_experiments.py
    ->
builds a list of Experiment objects
    ->
each Experiment has **name, task, method, script, command-line args**
    ->
runs each command as a separate subprocess
    ->
saves full output to experiments/logs/
    ->
parses metrics like dev acc and train loss
    ->
appends one row per run to experiments/results.csv


full-model:
    train all GPT-2 parameters

last-linear-layer:
    freeze GPT-2
    train only classification head

LoRA:
    freeze base GPT-2
    insert LoRA adapters
    train only LoRA params (+ task head where applicable)

ReFT:
    currently only a placeholder flag

---

# Default behavior

If you run the script without arguments:

```bash
python3 run_experiments.py
```

it is equivalent to running:

```bash
python3 run_experiments.py \
  --experiment comparison \
  --task all \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-5 \
  --temperature 1.2 \
  --top_p 0.9 \
  --lora_alpha 1.0 \
  --lora_target_modules query value
```

By default, it does **not** use CUDA. ```bash
--use_gpu
```

So the default run uses CPU.

---

## Default experiments

The default experiment group is:

```bash
--experiment comparison
```
This launches the main comparison experiments across all tasks.

With no arguments, the planned experiments are:

```text
sentiment_last_linear_layer
sentiment_full_model
sentiment_lora_r4
sentiment_reft_placeholder
paraphrase_full_model
paraphrase_lora_r4
paraphrase_reft_placeholder
sonnet_full_model
sonnet_lora_r4
sonnet_reft_placeholder
```
So by default, `run_experiments.py` plans **10 experiments**.

**THIS INCLUDES PARAPHRASE EXPERIMENTS ON THE DATASET SO TIME MIGHT SKYROCKET!!!**

---

## Default task

The default task is:

```bash
--task all
```

This means the default comparison includes:

```text
sentiment
paraphrase
sonnet
```

To restrict the run to only one task, use:

```bash
--task sentiment
```

or:

```bash
--task paraphrase
```

or:

```bash
--task sonnet
```

---

## Default training values

Default training values are:

```text
epochs      = 1
batch_size  = 8
lr          = 1e-5
use_gpu     = False
```

So every launched task script receives:

```bash
--epochs 1 --batch_size 8 --lr 1e-5
```

unless you override them.

Example override:

```bash
python3 run_experiments.py \
  --experiment comparison \
  --task sonnet \
  --epochs 3 \
  --batch_size 4 \
  --lr 1e-4
```

---

## Default LoRA values

Default LoRA settings are:

```text
lora_alpha          = 1.0
lora_target_modules = query value
lora_ranks          = 2 4 8 16
```

For the normal comparison experiment, LoRA uses rank 4 by default:

```text
sentiment_lora_r4
paraphrase_lora_r4
sonnet_lora_r4
```

Those commands use:

```bash
--enable_lora --lora_r 4 --lora_alpha 1.0 --lora_target_modules query value
```

The full `--lora_ranks 2 4 8 16` list is only used when running:

```bash
python3 run_experiments.py --experiment lora_rank
```

---

## Default ReFT behavior

ReFT is included in the planned comparisons as a placeholder:

```text
sentiment_reft_placeholder
paraphrase_reft_placeholder
sonnet_reft_placeholder
```

These commands use:

```bash
--enable_reft
```

At the moment, ReFT does not do anything unless the ReFT implementation is added to the task scripts.

---

## Default sonnet generation values

Default sonnet generation values are:

```text
temperature = 1.2
top_p       = 0.9
```

So sonnet experiments receive:

```bash
--temperature 1.2 --top_p 0.9
```

Currently implemented in `sonnet_generation.py`:

```text
temperature sampling
top-p / nucleus sampling
EOS stopping
max_length stopping
```

Planned but not yet implemented:

```text
top-k sampling
line-count stopping
beam search
```

---

## Default sweep values

These are only used when running specific experiment groups.

For LoRA rank ablation:

```bash
python3 run_experiments.py --experiment lora_rank
```

the default ranks are:

```text
2 4 8 16
```

For limited-data experiments:

```bash
python3 run_experiments.py --experiment limited_data
```

the default training fractions are:

```text
0.10 0.25 0.50 1.00
```

For sonnet decoding sweeps:

```bash
python3 run_experiments.py --experiment sonnet_decoding
```

the default sweeps are:

```text
temperature_sweep = 0.7 1.0 1.2
top_p_sweep       = 0.8 0.9 0.95
top_k_sweep       = 20 50 100
```

Note: `top_k_sweep` needs `--top_k` to be implemented in `sonnet_generation.py` before it works as a real experiment.

---

# Explanation of main arguments

## `--experiment`

Chooses which experiment group to run.

Example:

```bash
python3 run_experiments.py --experiment comparison
```

Options:

- `comparison`: compare full finetuning, last-layer, LoRA, and ReFT placeholder.
- `lora_rank`: run LoRA rank ablations.
- `limited_data`: run experiments with different training-data fractions.
- `sonnet_decoding`: run sonnet generation decoding sweeps.
- `all`: run every experiment group.

---

## `--task`

Chooses which task to run.

Example:

```bash
python3 run_experiments.py --experiment comparison --task sonnet
```

Options:

- `all`: run all tasks.
- `sentiment`: run only SST/CFIMDB classifier experiments.
- `paraphrase`: run only Quora paraphrase experiments.
- `sonnet`: run only sonnet generation experiments.

---

## `--dry-run`

Prints planned commands without running training.

Example:

```bash
python3 run_experiments.py --experiment all --dry-run
```

Use this before launching long experiments.

---

## `--max-runs`

Limits how many experiments are launched after filtering by `--experiment` and `--task`.

Example:

```bash
python3 run_experiments.py --experiment comparison --task sonnet --max-runs 1
```

The matching sonnet comparison experiments are ordered like this:

```text
1. sonnet_full_model
2. sonnet_lora_r4
3. sonnet_reft_placeholder
```

So this command runs only:

```text
sonnet_full_model
```

If you run:

```bash
python3 run_experiments.py --experiment comparison --task sonnet --max-runs 2
```

it runs:

```text
sonnet_full_model
sonnet_lora_r4
```

You can always confirm what will run using:

```bash
python3 run_experiments.py --experiment comparison --task sonnet --max-runs 2 --dry-run
```

---

## `--epochs`

Sets the number of training epochs.

Example:

```bash
python3 run_experiments.py --experiment comparison --task sonnet --epochs 1
```
---

## `--batch_size`

Sets the batch size passed to the training script.

Example:

```bash
python3 run_experiments.py --experiment comparison --batch_size 8
```
---

## `--lr`

Sets the learning rate.

Example:

```bash
python3 run_experiments.py --experiment comparison --lr 1e-5
```
---

## `--use_gpu`

Runs task scripts with CUDA enabled.

Example:

```bash
python3 run_experiments.py --experiment comparison --use_gpu
```

---

# LoRA arguments

## `--lora_ranks`

Sets the LoRA ranks used in the LoRA rank ablation.

Default:

```text
2 4 8 16
```

Example:

```bash
python3 run_experiments.py --experiment lora_rank --lora_ranks 2 4 8
```

Run only rank 4 on sonnets:

```bash
python3 run_experiments.py \
  --experiment lora_rank \
  --task sonnet \
  --lora_ranks 4
```

---

## `--lora_alpha`

Sets the LoRA scaling alpha.

Example:

```bash
python3 run_experiments.py --experiment comparison --lora_alpha 8.0
```

The LoRA layer uses:

```text
scaling = alpha / rank
```

So increasing `lora_alpha` increases the strength of the low-rank update.

---

## `--lora_target_modules`

Chooses which GPT-2 modules receive LoRA adapters.

Available options:

```text
query
key
value
dense
```

Default:

```text
query value
```

Example:

```bash
python3 run_experiments.py \
  --experiment comparison \
  --lora_target_modules query value
```

Use LoRA on query, key, value, and the attention output dense layer:

```bash
python3 run_experiments.py \
  --experiment comparison \
  --lora_target_modules query key value dense
```

---

# Sonnet decoding arguments

These are used for sonnet generation experiments.

## `--temperature`

Controls randomness during generation.

Example:

```bash
python3 run_experiments.py --experiment comparison --task sonnet --temperature 1.2
```

Lower values are more conservative:

```text
0.7
```

Higher values are more random:

```text
1.2
```

---

## `--top_p`

Controls nucleus sampling.

Example:

```bash
python3 run_experiments.py --experiment comparison --task sonnet --top_p 0.9
```

Common values:

```text
0.8
0.9
0.95
```

---

## `--temperature_sweep`

Sets temperatures used in sonnet decoding experiments.

Example:

```bash
python3 run_experiments.py \
  --experiment sonnet_decoding \
  --task sonnet \
  --temperature_sweep 0.7 1.0 1.2
```

---

## `--top_p_sweep`

Sets top-p values used in sonnet decoding experiments.

Example:

```bash
python3 run_experiments.py \
  --experiment sonnet_decoding \
  --task sonnet \
  --top_p_sweep 0.8 0.9 0.95
```

---

## `--top_k_sweep`

Sets top-k values used in sonnet decoding experiments.

Example:

```bash
python3 run_experiments.py \
  --experiment sonnet_decoding \
  --task sonnet \
  --top_k_sweep 20 50 100
```

Note: `sonnet_generation.py` must support `--top_k` before this works as a real experiment.

---

# Limited-data arguments

## `--train_fractions`

Sets what fraction of the training data to use.

Default:

```text
0.10 0.25 0.50 1.00
```

Example:

```bash
python3 run_experiments.py \
  --experiment limited_data \
  --train_fractions 0.10 0.25 0.50 1.00
```

Run only sonnet limited-data experiments:

```bash
python3 run_experiments.py \
  --experiment limited_data \
  --task sonnet \
  --train_fractions 0.10 0.25 0.50 1.00
```

Note: the task scripts must support `--train_fraction` before these real runs work.

Until then, use dry-run mode:

```bash
python3 run_experiments.py --experiment limited_data --dry-run
```

---

## Fast full-model vs LoRA sonnet comparison

```bash
python3 run_experiments.py \
  --experiment comparison \
  --task sonnet \
  --epochs 1 \
  --batch_size 8 \
  --max-runs 2
```

This runs:

```text
sonnet_full_model
sonnet_lora_r4
```

---

# Dry-runs to check that everything works

Dry-run main comparison:

```bash
python3 run_experiments.py --experiment comparison --dry-run
```

Dry-run all experiments:

```bash
python3 run_experiments.py --experiment all --dry-run
```

Dry-run LoRA rank ablation:

```bash
python3 run_experiments.py --experiment lora_rank --dry-run
```

Dry-run limited-data experiments:

```bash
python3 run_experiments.py --experiment limited_data --dry-run
```

Dry-run sonnet decoding sweeps:

```bash
python3 run_experiments.py --experiment sonnet_decoding --dry-run
```

---

# Example experiment commands

## Compare full-model, LoRA, and ReFT placeholder on sonnets

```bash
python3 run_experiments.py \
  --experiment comparison \
  --task sonnet \
  --epochs 1 \
  --batch_size 8
```

---

## Run LoRA rank ablation only on sonnets

```bash
python3 run_experiments.py \
  --experiment lora_rank \
  --task sonnet \
  --epochs 1 \
  --batch_size 8 \
  --lora_ranks 2 4 8 16
```

---

## Run LoRA with all target modules

```bash
python3 run_experiments.py \
  --experiment lora_rank \
  --task sonnet \
  --epochs 1 \
  --batch_size 8 \
  --lora_ranks 4 \
  --lora_target_modules query key value dense
```

---

## Run a sonnet temperature sweep

```bash
python3 run_experiments.py \
  --experiment sonnet_decoding \
  --task sonnet \
  --epochs 1 \
  --batch_size 8 \
  --temperature_sweep 0.7 1.0 1.2
```

---

## Run a top-p sweep

```bash
python3 run_experiments.py \
  --experiment sonnet_decoding \
  --task sonnet \
  --epochs 1 \
  --batch_size 8 \
  --top_p_sweep 0.8 0.9 0.95
```

---

# Reading results

After running experiments, check:

```bash
cat experiments/results.csv
```

The `results.csv` file contains:

```text
timestamp
name
group
task
method
status
returncode
duration_sec
dev_acc
train_loss
trainable_percent
command
log_file
notes
```

Important columns:

- `name`: experiment name.
- `task`: sentiment, paraphrase, or sonnet.
- `method`: full-model, last-linear-layer, LoRA, or ReFT.
- `status`: ok, failed, or dry-run.
- `returncode`: process return code. `0` means success.
- `duration_sec`: runtime.
- `dev_acc`: dev accuracy when available.
- `train_loss`: final detected training loss.
- `trainable_percent`: percentage of trainable parameters if printed by LoRA.
- `command`: exact command that was launched.
- `log_file`: full terminal output for that experiment.
- `notes`: extra notes, such as whether ReFT is only a placeholder.

---

# Logs

Logs are stored in:

```text
experiments/logs/
```
---

# Plotting results

After having data, generate plots only:

```bash
python3 run_experiments.py --plot-only
```

Plots are saved in:

```text
experiments/plots/
```

---

# TLDR

- Running without arguments launches the default comparison over all tasks:

```bash
python3 run_experiments.py
```

- Always check first with:

```bash
python3 run_experiments.py --dry-run
```

- LoRA is enabled with:

```bash
--enable_lora
```

- LoRA rank is passed to task scripts as:

```bash
--lora_r
```

- ReFT is assumed to be enabled with:

```bash
--enable_reft
```

- ReFT is currently only a placeholder until the ReFT module is implemented.
- Limited-data experiments require `--train_fraction` to be implemented in each task script.
- Sonnet top-k decoding requires `--top_k` to be implemented in `sonnet_generation.py`.
- Sonnet line-count stopping requires `--line_count_stopping` to be implemented in `sonnet_generation.py`.


