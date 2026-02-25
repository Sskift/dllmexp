import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import argparse
import json
import logging
from copy import deepcopy
import torch
from transformers import AutoModel, AutoTokenizer
from lm_eval import evaluator
from harness import DreamEvalHarness, LladaEvalHarness
from eval_config.monkey_patch import apply_custom_task_configs
from dream import generation_utils as dream_gen
from utils import parse_results
from dream.modeling_dream import DreamModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TASKS = {
    "humaneval": "humaneval",
    "mbpp": "mbpp",
    "gsm8k": "gsm8k",
    # Use generation config for TruthfulQA to avoid log-likelihood-only MC scoring on diffusion models
    "truthfulqa": "truthfulqa",
}


MODEL_DEFAULTS = {
    "dream": {
        "truthfulqa": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
        "humaneval": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
        "mbpp": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
        "gsm8k": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
        "default": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
    },
    "diffucoder": {
        "truthfulqa": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
        "humaneval": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
        "mbpp": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
        "gsm8k": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
        "default": {"steps": 256, "temperature": 0.0, "top_p": 0.95, "max_new_tokens": 256, "alg": "maskgit_plus", "tokens_per_step": 1},
    },
    "llada": {
        "truthfulqa": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "tokens_per_step": 1},
        "humaneval": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "random", "tokens_per_step": 1},
        "mbpp": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "tokens_per_step": 1},
        "gsm8k": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "tokens_per_step": 1},
        "default": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "tokens_per_step": 1},
    },
    "llada1.5": {
        "truthfulqa": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "tokens_per_step": 1},
        "humaneval": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "tokens_per_step": 1},
        "mbpp": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "tokens_per_step": 1},
        "gsm8k": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "tokens_per_step": 1},
        "default": {"alg": "low_confidence", "num_steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "remasking": "low_confidence", "tokens_per_step": 1},
    },
}


ALLOWED_ALGS = {
    "dream": {"maskgit_plus"},
    "diffucoder": {"maskgit_plus"},
    "llada": {"low_confidence", "random", "leftright","ddola","sdola"},
    "llada1.5": {"low_confidence", "random", "leftright"},
}


def _template_llada(text: str) -> str:
    return f"""
    <|startoftext|><|start_header_id|>user<|end_header_id|>

    You are an intelligent programming assistant to produce Python algorithmic solutions. Can you complete the following Python function?
    ```python
    {text}
    ```
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ```python
    {text}
    """


def _template_chat_code(text: str) -> str:
    return f"""<|im_start|>system
    You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
    <|im_start|>user
    Can you complete the following Python function?{text}<|im_end|>
    <|im_start|>assistant
    ```python
    {text}
    """


def _template_chat_code_mbpp(text: str) -> str:
    return f"""<|im_start|>user
    You are an expert Python programmer, and here is your task: {text}
    <|im_end|>
    <|im_start|>assistant
    ```python
    """


def _template_chat_reasoning(text: str) -> str:
    return f"""<|im_start|>system
    You are a careful reasoning assistant. Answer the question briefly and factually.<|im_end|>
    <|im_start|>user
    {text}
    <|im_end|>
    <|im_start|>assistant
    """


def get_prompt_template(model_alias: str, task_name: str):
    if task_name in {"humaneval", "mbpp"}:
        if model_alias.startswith("llada"):
            return _template_llada
        if model_alias == "dream" and task_name == "humaneval":
             # For Dream on HumanEval, avoid pre-filling the signature to prevent early stopping
             return _template_chat_code_mbpp
        return _template_chat_code_mbpp if task_name == "mbpp" else _template_chat_code
    if task_name in {"truthfulqa", "gsm8k"}:
        return _template_chat_reasoning
    return None


def _merge_generation_config(model_alias: str, task: str, overrides: dict) -> dict:
    defaults = MODEL_DEFAULTS[model_alias].get(task, MODEL_DEFAULTS[model_alias]["default"])
    config = deepcopy(defaults)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def get_model(args):
    model_alias = args.model_alias
    task_name = args.task

    logger.info(f"Loading model: {model_alias} for task: {task_name}")

    allowed = ALLOWED_ALGS.get(model_alias)
    if allowed is not None and args.alg is not None and args.alg not in allowed:
        raise ValueError(f"Invalid alg '{args.alg}' for model '{model_alias}'. Allowed: {sorted(allowed)}")

    prompt_template = get_prompt_template(model_alias, task_name)

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if model_alias == "dream":
        overrides = {
            "steps": args.num_steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "alg": args.alg,
            "tokens_per_step": args.tokens_per_step,
        }
        cfg = _merge_generation_config("dream", task_name, overrides)
        dream = DreamModel.from_pretrained(
            args.dream_ckpt,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
            device_map=("auto" if args.device == "auto" else (args.device if args.device == "cpu" else "cuda")),
        )
        tokenizer = AutoTokenizer.from_pretrained(args.dream_ckpt, trust_remote_code=True)
        harness = DreamEvalHarness(
            pretrained=dream,
            tokenizer=tokenizer,
            steps=cfg["steps"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            alg=cfg["alg"],
            max_new_tokens=cfg["max_new_tokens"],
            tokens_per_step=cfg.get("tokens_per_step"),
            prompt_template=prompt_template,
        )
        harness.is_code_task = task_name in {"humaneval", "mbpp"}
        return harness

    if model_alias == "diffucoder":
        overrides = {
            "steps": args.num_steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "alg": args.alg,
            "tokens_per_step": args.tokens_per_step,
        }
        cfg = _merge_generation_config("diffucoder", task_name, overrides)
        model = AutoModel.from_pretrained(
            args.diffucoder_ckpt,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if args.device == "auto" else None,
        )
        if args.device != "auto":
            model = model.to(args.device)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.diffucoder_ckpt, trust_remote_code=True)

        # Patch DiffuCoder generation to use Dream's generation mixin (tokens_per_step, sane max_length)
        def _bind(obj, fn):
            return fn.__get__(obj, obj.__class__)

        # Bind core mixin methods
        for name in [
            "_prepare_generation_config",
            "_prepare_special_tokens",
            "_prepare_generated_length",
            "_validate_generated_length",
        ]:
            if hasattr(dream_gen.DreamGenerationMixin, name):
                setattr(model, name, _bind(model, getattr(dream_gen.DreamGenerationMixin, name)))

        model.generation_config = dream_gen.DreamGenerationConfig.from_model_config(model.config)
        model._diffusion_generate_patched = _bind(model, dream_gen.DreamGenerationMixin.diffusion_generate)

        def _dream_diffusion_generate(self, inputs=None, attention_mask=None, **kwargs):
            cfg = dream_gen.DreamGenerationConfig.from_model_config(self.config)
            for key, val in kwargs.items():
                if hasattr(cfg, key) and val is not None:
                    setattr(cfg, key, val)
            if cfg.max_new_tokens is not None:
                cfg.max_length = (inputs.shape[1] if inputs is not None else 0) + cfg.max_new_tokens
            cfg.return_dict_in_generate = True
            cfg.output_history = False
            return self._diffusion_generate_patched(
                inputs=inputs,
                attention_mask=attention_mask,
                generation_config=cfg,
            )

        model.diffusion_generate = _dream_diffusion_generate.__get__(model, model.__class__)

        harness = DreamEvalHarness(
            pretrained=model,
            tokenizer=tokenizer,
            steps=cfg["steps"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            alg=cfg["alg"],
            max_new_tokens=cfg["max_new_tokens"],
            tokens_per_step=cfg.get("tokens_per_step"),
            prompt_template=prompt_template,
        )
        harness.model_alias = "diffucoder"
        harness.is_code_task = task_name in {"humaneval", "mbpp"}
        return harness

    if model_alias == "llada":
        overrides = {
            "num_steps": args.num_steps,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "remasking": args.remasking,
            "tokens_per_step": args.tokens_per_step,
            "alg": args.alg,
            "cfg_scale": args.cfg_scale,
        }
        cfg = _merge_generation_config("llada", task_name, overrides)
        llada = AutoModel.from_pretrained(
            args.llada_ckpt,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if args.device == "auto" else None,
        )
        if args.device != "auto":
            llada = llada.to(args.device)
        llada = llada.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.llada_ckpt, trust_remote_code=True)
        harness = LladaEvalHarness(
            pretrained=llada,
            tokenizer=tokenizer,
            alg=cfg["alg"],
            num_steps=cfg["num_steps"],
            gen_length=cfg["gen_length"],
            block_length=cfg["block_length"],
            temperature=cfg["temperature"],
            remasking=cfg["remasking"],
            tokens_per_step=cfg.get("tokens_per_step", 1),
            cfg_scale=cfg.get("cfg_scale", 0.0),
            prompt_template=prompt_template,
        )
        harness.is_code_task = task_name in {"humaneval", "mbpp"}
        return harness

    if model_alias == "llada1.5":
        overrides = {
            "num_steps": args.num_steps,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "remasking": args.remasking,
            "tokens_per_step": args.tokens_per_step,
            "alg": args.alg,
            "cfg_scale": args.cfg_scale,
        }
        cfg = _merge_generation_config("llada1.5", task_name, overrides)
        llada = AutoModel.from_pretrained(
            args.llada15_ckpt,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if args.device == "auto" else None,
        )
        if args.device != "auto":
            llada = llada.to(args.device)
        llada = llada.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.llada15_ckpt, trust_remote_code=True)
        harness = LladaEvalHarness(
            pretrained=llada,
            tokenizer=tokenizer,
            alg=cfg["alg"],
            num_steps=cfg["num_steps"],
            gen_length=cfg["gen_length"],
            block_length=cfg["block_length"],
            temperature=cfg["temperature"],
            remasking=cfg["remasking"],
            tokens_per_step=cfg.get("tokens_per_step", 1),
            cfg_scale=cfg.get("cfg_scale", 0.0),
            prompt_template=prompt_template,
        )
        harness.model_alias = "llada1.5"
        harness.is_code_task = task_name in {"humaneval", "mbpp"}
        return harness

    raise ValueError(f"Unknown model alias: {model_alias}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion-based code models on common benchmarks.")
    parser.add_argument("--model_alias", required=True, choices=["dream", "llada", "llada1.5", "diffucoder"], help="Model to evaluate")
    parser.add_argument("--task", required=True, choices=["humaneval", "mbpp", "gsm8k", "truthfulqa"], help="Benchmark task")
    parser.add_argument("--output_dir", default="results", help="Directory to save evaluation outputs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick tests")
    parser.add_argument("--alg", type=str, default=None, help="Decoding algorithm override; must be in per-model whitelist")
    parser.add_argument("--num_steps", type=int, default=None, help="Diffusion steps override")
    parser.add_argument("--gen_length", type=int, default=None, help="Generation length for LLaDA-style decoding")
    parser.add_argument("--block_length", type=int, default=None, help="Block length for LLaDA-style decoding")
    parser.add_argument("--tokens_per_step", type=int, default=None, help="Tokens per step for left-right decoding")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p override")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Maximum generated tokens override")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Classifier-free guidance scale for LLaDA")
    parser.add_argument("--remasking", type=str, default=None, help="Remasking strategy for LLaDA diffusion decoding")
    parser.add_argument("--dream_ckpt", type=str, default="Dream-org/Dream-v0-Instruct-7B", help="Dream checkpoint")
    parser.add_argument("--llada_ckpt", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="LLaDA checkpoint")
    parser.add_argument("--llada15_ckpt", type=str, default="GSAI-ML/LLaDA-1.5", help="LLaDA 1.5 checkpoint")
    parser.add_argument("--diffucoder_ckpt", type=str, default="apple/DiffuCoder-7B-cpGRPO", help="DiffuCoder checkpoint")
    parser.add_argument("--tag", type=str, default="", help="Optional tag appended to output filename")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Torch dtype for model weights")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"], help="Device to place model on (use auto for hf accelerate device_map)")

    args = parser.parse_args()

    # Apply custom task YAML overrides (hendrycks_math, gpqa, etc.) if present
    apply_custom_task_configs()

    model = get_model(args)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    task_label = args.task
    if args.alg:
        task_label += f"_{args.alg}"
    if args.cfg_scale is not None:
        task_label += f"_cfg{args.cfg_scale}"
    if args.limit is not None:
        task_label += f"_limit{args.limit}"
    if args.num_steps is not None:
        task_label += f"_steps={args.num_steps}"
    if args.max_new_tokens is not None:
        task_label += f"_gen={args.max_new_tokens}"
    if args.tokens_per_step is not None:
        task_label += f"_tps={args.tokens_per_step}"
    if args.tag:
        task_label += f"_{args.tag}"

    output_filename = f"{args.model_alias}_{task_label}.json"
    output_path = os.path.join(output_dir, output_filename)
    logger.info(f"Results will be written to {output_path}")

    # Setup job-level cache to resume partial progress
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{output_filename}.jsonl")
    if hasattr(model, "load_cache"):
        logger.info(f"Using generation cache: {cache_path}")
        model.load_cache(cache_path)

    if args.task in {"humaneval", "mbpp"}:
        system_instruction = "You are an expert Python coding assistant. Write complete, executable solutions; reply with code blocks only unless the task explicitly asks for a short answer."
    else:
        system_instruction = "You are a careful reasoning assistant. Solve the problem step by step and give a concise final answer."
    if args.task == "truthfulqa" and getattr(model, "model_alias", None) in {"dream", "diffucoder", "llada", "llada1.5"}:
        # Avoid loglikelihood-based MC scoring on diffusion models; use generation-only variant instead
        task_names = ["truthfulqa_gen"]
        logger.info("Using truthfulqa_gen only for diffusion-style model to rely on generation scoring.")
    else:
        task_names = [TASKS[args.task]]

    results = evaluator.simple_evaluate(  # type: ignore[name-defined]
        model=model,
        tasks=task_names,
        batch_size=1,
        limit=args.limit,
        log_samples=True,
        write_out=True,
        num_fewshot=0,
        apply_chat_template=False,
        system_instruction=system_instruction,
        confirm_run_unsafe_code=True,
    )

    results["profile"] = model.get_profile()
    parsed_results = parse_results(results, task_name=args.task)

    with open(output_path, "w") as f:
        json.dump(parsed_results, f, indent=4)


if __name__ == "__main__":
    main()
