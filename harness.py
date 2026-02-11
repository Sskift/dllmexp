import math
import time
from typing import Optional

import numpy as np
import torch
from lm_eval.models.huggingface import HFLM
from llada.llada_generate import llada_ar_generate, llada_diffusion_generate
from diffucoder.generate import diffucoder_generate


class _ProfilingHarness(HFLM):
    def __init__(self, *args, prompt_template=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile: dict[str, list[float]] = {}
        self.prompt_template = prompt_template

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
        else:
            outputs = llada_diffusion_generate(
                self.model,
                context,
                num_steps=num_steps,
                gen_length=gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=0.0,
                remasking=self.alg if self.alg in {"low_confidence", "random"} else self.remasking,
            )

        end = time.time()

        generated_tokens = 0
        for token_id in outputs[0][context.shape[1]:]:
            generated_tokens += 1
            if token_id == self.tokenizer.eos_token_id:
                break

        self._log_profile(generated_tokens, end - start)
        return outputs


class Llada15EvalHarness(LladaEvalHarness):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_alias = "llada1.5"


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
        prompt_template=None,
    ) -> None:
        super().__init__(pretrained=pretrained, tokenizer=tokenizer, prompt_template=prompt_template)
        self.model_alias = "dream"
        self.steps = steps
        self.temperature = temperature
        self.top_p = top_p
        self.alg = alg
        self.max_new_tokens = max_new_tokens

    @property
    def max_gen_toks(self) -> int:
        return self.max_new_tokens

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # stop is intentionally ignored to avoid premature truncation from lm-eval stop sequences
        context, attention_mask = self._prepare_context(context)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_new_tokens = max_length - context.shape[1] if max_length is not None else self.max_new_tokens
        start = time.time()
        outputs = diffucoder_generate(
            self.model,
            context,
            attention_mask=attention_mask,
            steps=self.steps,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=max_new_tokens,
            alg=self.alg,
        )
        end = time.time()

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated_tokens = sequences.shape[1] - context.shape[1]
        self._log_profile(generated_tokens, end - start)
        return sequences


class DiffuCoderEvalHarness(_ProfilingHarness):
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
        prompt_template=None,
    ) -> None:
        super().__init__(pretrained=pretrained, tokenizer=tokenizer, prompt_template=prompt_template)
        self.model_alias = "diffucoder"
        self.steps = steps
        self.temperature = temperature
        self.top_p = top_p
        self.alg = alg
        self.max_new_tokens = max_new_tokens

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
            pad_token_id=pad_token_id,
            steps=self.steps,
            temperature=self.temperature,
            top_p=self.top_p,
            alg=self.alg,
            alg_temp=0.0,
            max_new_tokens=max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
        )
        end = time.time()

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated_tokens = sequences.shape[1] - context.shape[1]
        self._log_profile(generated_tokens, end - start)
        return sequences
