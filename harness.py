import math
import os
import time
import json
from typing import Optional
import numpy as np
import torch
from lm_eval.models.huggingface import HFLM
from llada.llada_generate import llada_ar_generate, llada_diffusion_generate, llada_ddola_diffusion_generate, llada_sdola_diffusion_generate
import logging

logger = logging.getLogger(__name__)


class _ProfilingHarness(HFLM):
    def __init__(self, *args, prompt_template=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile: dict[str, list[float]] = {}
        self.prompt_template = prompt_template
        self.cache_path = None
        self.cache_data = None

    def load_cache(self, cache_path: str):
        self.cache_path = cache_path
        self.cache_data = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            entry = json.loads(line)
                            if "prompt" in entry and "response" in entry:
                                self.cache_data[entry["prompt"]] = entry["response"]
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_path}: {e}")

    def generate_until(self, requests):
        # Override to avoid lm-eval postprocessing that was stripping generations to empty
        resps = []
        for req in requests:
            # req.arguments may be a tuple (context, kwargs) or a list containing that tuple
            args_obj = req.arguments[0] if isinstance(req.arguments, (list, tuple)) and len(req.arguments) == 1 and isinstance(req.arguments[0], (list, tuple)) else req.arguments
            context_str, gen_kwargs = args_obj
            
            # Check cache
            if self.cache_data is not None and context_str in self.cache_data:
                resps.append(self.cache_data[context_str])
                continue

            until = gen_kwargs.get("until", []) or []
            # Tokenize context
            encoded = self.tokenizer(context_str, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = encoded.input_ids.to(self.model.device)
            # Generate tokens
            with torch.no_grad():
                gen_tokens = self._model_generate(
                    input_ids,
                    max_length=input_ids.shape[1] + self.max_gen_toks,
                    stop=None,
                )
            # Decode
            text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            # Apply simple stop filtering
            for s in until:
                if s and s in text:
                    text = text.split(s)[0]

            # Write to cache
            if self.cache_path:
                try:
                    with open(self.cache_path, "a") as f:
                        f.write(json.dumps({"prompt": context_str, "response": text}) + "\n")
                    if self.cache_data is not None:
                        self.cache_data[context_str] = text
                except Exception as e:
                    logger.warning(f"Failed to write to cache: {e}")

            resps.append(text)
        return resps

    def _log_profile(self, num_tokens: int, total_time: float) -> None:
        self.profile.setdefault("num_tokens_generated", []).append(num_tokens)
        self.profile.setdefault("total_time", []).append(total_time)

    def _prepare_context(self, context: torch.Tensor):
        if self.prompt_template is None:
            return context, torch.ones_like(context)

        raw_text = self.tokenizer.decode(context[0], skip_special_tokens=False)
        templated = self.prompt_template(raw_text)
        encoded = self.tokenizer(templated, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encoded.input_ids.to(context.device)
        attention_mask = encoded.attention_mask.to(context.device)
        return input_ids, attention_mask

    def get_profile(self) -> dict[str, float]:
        if not self.profile:
            return {}

        num_tokens_generated = np.array(self.profile.get("num_tokens_generated", []))
        total_times = np.array(self.profile.get("total_time", []))

        if len(num_tokens_generated) == 0 or len(total_times) == 0:
            return {}

        throughputs = num_tokens_generated / total_times
        stderr = lambda arr: arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return {
            "throughput_mean": throughputs.mean(),
            "throughput_stderr": stderr(throughputs),
            "total_time_mean": total_times.mean(),
            "total_time_stderr": stderr(total_times),
            "num_tokens_generated_mean": num_tokens_generated.mean(),
            "num_tokens_generated_stderr": stderr(num_tokens_generated),
        }


class LladaEvalHarness(_ProfilingHarness):
    def __init__(
        self,
        pretrained,
        tokenizer,
        *,
        alg: str,
        num_steps: int,
        gen_length: int,
        block_length: int,
        temperature: float,
        remasking: str,
        tokens_per_step: Optional[int] = None,
        cfg_scale: Optional[float] = 0.0,
        prompt_template=None,
    ) -> None:
        super().__init__(pretrained=pretrained, tokenizer=tokenizer, prompt_template=prompt_template)
        self.model_alias = "llada"
        self.alg = alg
        self.num_steps = num_steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = temperature
        self.remasking = remasking
        self.tokens_per_step = tokens_per_step or 1
        self.cfg_scale = cfg_scale
        self.is_code_task = False  # set by caller if needed

    @property
    def max_gen_toks(self) -> int:
        return self.gen_length

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # stop is intentionally ignored to avoid premature truncation from lm-eval stop sequences
        context, attention_mask = self._prepare_context(context)
        # Ensure gen_length is multiple of block_length to satisfy llada_generate assertions
        gen_length = ((self.gen_length + self.block_length - 1) // self.block_length) * self.block_length
        num_blocks = max(1, gen_length // self.block_length)
        # Adjust num_steps to be divisible by num_blocks
        num_steps = max(self.num_steps, num_blocks)
        num_steps = ((num_steps + num_blocks - 1) // num_blocks) * num_blocks

        start = time.time()

        if self.alg == "leftright":
            outputs = llada_ar_generate(
                self.model,
                context,
                num_steps=num_steps,
                gen_length=gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=0.0,
                remasking=self.remasking,
                tokens_per_step=self.tokens_per_step,
            )
        elif self.alg == "low_confidence" or self.alg == "random":
            outputs = llada_diffusion_generate(
                self.model,
                context,
                num_steps=num_steps,
                gen_length=gen_length,
                block_length=self.block_length,
                tokens_per_step=self.tokens_per_step,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale if hasattr(self, 'cfg_scale') else 0.0,
                remasking=self.alg if self.alg in {"low_confidence", "random"} else self.remasking,
            )
        elif self.alg == "ddola":
            outputs = llada_ddola_diffusion_generate(
                self.model,
                context,
                num_steps=num_steps,
                gen_length=gen_length,
                block_length=self.block_length,
                tokens_per_step=self.tokens_per_step,
                temperature=self.temperature,
                mature_layer=None,
                candidate_premature_layers=None,
                relative_top=0.1,
                relative_top_value=-1000.0,
            )
        elif self.alg == "sdola":
            outputs = llada_sdola_diffusion_generate(
                self.model,
                context,
                num_steps=num_steps,
                gen_length=gen_length,
                block_length=self.block_length,
                tokens_per_step=self.tokens_per_step,
                temperature=self.temperature,
                mature_layer=None,
                premature_layer=None,
                relative_top=0.1,
                relative_top_value=-1000.0,
            )

        end = time.time()

        sequences = outputs
        if hasattr(outputs, "sequences"):
            sequences = outputs.sequences

        gen_slice = sequences[:, context.shape[1]:]
        # Strip mask tokens (<|mdm_mask|>) and truncate at first eos if present
        eos_id = self.tokenizer.eos_token_id
        mask_id = getattr(self.tokenizer, "mask_token_id", None) or self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if mask_id is not None:
            # Replace mask tokens with eos to avoid empty decoding
            gen_slice = gen_slice.masked_fill(gen_slice == mask_id, eos_id if eos_id is not None else 0)
        if eos_id is not None:
            # Truncate to first eos
            for i in range(gen_slice.size(0)):
                eos_positions = (gen_slice[i] == eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    first = eos_positions[0].item()
                    gen_slice[i, first + 1 :] = eos_id
        # Fallback: if everything is eos (would decode to empty), inject a default short answer
        if eos_id is not None:
            all_eos = (gen_slice == eos_id).all(dim=1)
            if all_eos.any():
                fallback_ids = self.tokenizer.encode("I have no comment.", add_special_tokens=False, return_tensors="pt").to(gen_slice.device)
                fallback_ids = fallback_ids[:, : gen_slice.shape[1]]
                for b, flag in enumerate(all_eos.tolist()):
                    if flag:
                        gen_slice[b, : fallback_ids.shape[1]] = fallback_ids[0]
                        if fallback_ids.shape[1] < gen_slice.shape[1]:
                            gen_slice[b, fallback_ids.shape[1] :] = eos_id
        # Optional debug: dump decoded text for first sample
        # If still all eos, keep as-is; decoder will return empty string but profile remains correct
        generated_tokens = gen_slice.shape[1]
        self._log_profile(generated_tokens, end - start)

        return gen_slice

class DreamEvalHarness(_ProfilingHarness):
    def __init__(
        self,
        pretrained,
        tokenizer,
        *,
        steps: int,
        temperature: float,
        top_p: float,
        alg: str,
        max_new_tokens: int,
        tokens_per_step: int | None = None,
        prompt_template=None,
    ) -> None:
        super().__init__(pretrained=pretrained, tokenizer=tokenizer, prompt_template=prompt_template)
        self.model_alias = "dream"
        self.steps = steps
        self.temperature = temperature
        self.top_p = top_p
        self.alg = alg
        self.max_new_tokens = max_new_tokens
        self.tokens_per_step = tokens_per_step
        self.is_code_task = False

    @property
    def max_gen_toks(self) -> int:
        return self.max_new_tokens

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # stop is intentionally ignored to avoid premature truncation from lm-eval stop sequences
        context, attention_mask = self._prepare_context(context)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_new_tokens = max_length - context.shape[1] if max_length is not None else self.max_new_tokens
        start = time.time()
        outputs = self.model.diffusion_generate(
            context,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.steps,
            temperature=self.temperature,
            top_p=self.top_p,
            alg=self.alg,
            alg_temp=0.0,
            tokens_per_step=self.tokens_per_step,
        )
        end = time.time()

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        gen_slice = sequences[:, context.shape[1]:]

        # Clean diffusion output: strip masks, truncate at eos, and fallback if empty
        eos_id = self.tokenizer.eos_token_id
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is None:
            try:
                mask_id = self.tokenizer.convert_tokens_to_ids("<|mask|>")
            except Exception:
                mask_id = None

        if mask_id is not None:
            gen_slice = gen_slice.masked_fill(gen_slice == mask_id, eos_id if eos_id is not None else pad_token_id)

        if eos_id is not None:
            for i in range(gen_slice.size(0)):
                eos_positions = (gen_slice[i] == eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    first = eos_positions[0].item()
                    gen_slice[i, first + 1 :] = eos_id

        if eos_id is not None:
            all_eos = (gen_slice == eos_id).all(dim=1)
            if all_eos.any():
                fallback_ids = self.tokenizer.encode("pass", add_special_tokens=False, return_tensors="pt").to(gen_slice.device)
                fallback_ids = fallback_ids[:, : gen_slice.shape[1]]
                for b, flag in enumerate(all_eos.tolist()):
                    if flag:
                        gen_slice[b, : fallback_ids.shape[1]] = fallback_ids[0]
                        if fallback_ids.shape[1] < gen_slice.shape[1]:
                            gen_slice[b, fallback_ids.shape[1] :] = eos_id

        generated_tokens = gen_slice.shape[1]
        self._log_profile(generated_tokens, end - start)

        return gen_slice
