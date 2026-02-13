# dllmexp: Diffusion LLM Evaluation Framework

This repository provides an evaluation framework for diffusion-based Large Language Models (LLMs) and Code LLMs, including models like **Dream**, **DiffuCoder**, and **LLaDA**. It supports running various diffusion decoding algorithms across standard benchmarks.

## 1. Environment Setup

The codebase relies on Python and HuggingFace libraries.

### Conda Environment
It is recommended to use the `dllmexp` conda environment as referenced in the scripts.

```bash
conda create -n dllmexp python=3.10
conda activate dllmexp
```

### Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```
*Key dependencies:*
- `transformers==4.54.0`
- `lm-eval==0.4.8`
- `numpy>=1.24`
- `tqdm`
- `prettytable`

**Note:** Ensure `HF_HOME` and `HF_DATASETS_CACHE` are set if you need to store models in a specific directory (already handled in `scripts/exp.sh`).

## 2. Quick Start

### Single Evaluation
You can run a single evaluation job using `eval.py`. Here is an example showing all supported arguments:

```bash
python eval.py \
    # [Required] Model selection: dream, llada, llada1.5, diffucoder
    --model_alias llada \
    # [Required] Task selection: humaneval, mbpp, gsm8k, truthfulqa
    --task humaneval \
    # [Optional] Algorithm override (e.g., maskgit_plus, low_confidence, random)
    --alg low_confidence \
    # [Optional] decoding parameters
    --tokens_per_step 1 \
    --num_steps 128 \
    --gen_length 512 \
    --block_length 64 \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_new_tokens 512 \
    --remasking low_confidence \
    # [Optional] Job control
    --limit 10 \
    --output_dir results \
    --tag debug_run \
    --dtype bfloat16 \
    --device cuda \
    # [Optional] Model Checkpoint Overrides (defaults shown)
    --dream_ckpt "Dream-org/Dream-v0-Instruct-7B" \
    --llada_ckpt "GSAI-ML/LLaDA-8B-Instruct" \
    --llada15_ckpt "GSAI-ML/LLaDA-1.5" \
    --diffucoder_ckpt "apple/DiffuCoder-7B-cpGRPO"
```

### Batch Experiments
To run the full suite of 48 experiments (all models x all tasks x sweep over TPS), use the provided shell script. This script handles GPU scheduling and resumes from checkpoints automatically.

```bash
# Runs inside a tmux session
./scripts/exp.sh
```

Monitor progress with:
```bash
./scripts/progress.sh
```

## 3. Supported Configuration

### Models (`--model_alias`)
| Model Alias | Description | Checkpoint Default |
|-------------|-------------|--------------------|
| `dream` | Dream-v0-Instruct-7B | `Dream-org/Dream-v0-Instruct-7B` |
| `diffucoder`| DiffuCoder-7B | `apple/DiffuCoder-7B-cpGRPO` |
| `llada` | LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` |
| `llada1.5` | LLaDA-1.5 | `GSAI-ML/LLaDA-1.5` |

### Benchmarks (`--task`)
| Task Name | Description |
|-----------|-------------|
| `humaneval` | HumanEval Python Coding Benchmark |
| `mbpp` | MBPP Python Coding Benchmark |
| `gsm8k` | Grade School Math Reasoning |
| `truthfulqa`| TruthfulQA (Generation mode) |

### Algorithms (`--alg`)
- **Dream / DiffuCoder**: `maskgit_plus`
- **LLaDA**: `low_confidence`, `random`, `leftright`

### Common Parameters
- `--tokens_per_step`: Tokens generated per diffusion step (default: 1).
- `--num_steps`: Number of diffusion steps (or auto-calculated for LLaDA block decoding).
- `--temperature`: Sampling temperature.
- `--limit`: Limit number of samples (useful for debugging).
- `--output_dir`: Directory to save JSON results (default: `results`).

## 4. Performance & Caching & bash example
- **Job Resume**: `eval.py` automatically maintains a `results/cache/` directory. If a job crashes, re-running it will skip already generated samples.
- **Experiment Skip**: `scripts/exp.sh` detects if a final result file exists and skips listing that job entirely.
- **Explanation**: `scripts/exp.sh` shows how the experiments that the 4 models run on 4 benchmarks with tps=1,2,3, totally 48, are carried out in a tmux window. Before start, you need to set your own hf_home and hf_cache correctly.

## 5. Extending the Framework

### Adding a New Model
To add a new model, modify **`eval.py`**:
1.  **Defaults**: Add a configuration dictionary to `MODEL_DEFAULTS` defining default steps, algorithms, etc.
2.  **Loader**: Update the `get_model()` function to handle the new `model_alias` and initialize the appropriate Harness.
3.  **Harness**: If special generation logic is needed, implement a new harness class in `harness.py` inheriting from `_ProfilingHarness` or `HFLM`.

### Adding a New Benchmark
To add a new benchmark, modify **`eval.py`**:
1.  **Mapping**: Update the `TASKS` dictionary to map your CLI task name to the `lm-eval` task name.
2.  **Templates**: If the task requires a specific chat template, update `get_prompt_template`.
3.  **Pattern Recognition**: You may need to modify **`utils.py`** to adjust the output to the formay you want.

## 6. Declaration

1. This repo is forked from apd(https://github.com/danielmisrael/apd) and restructed on it.
2. Ackonwledge for gemini 3.0 pro and codex 5.1 