import torch


def diffucoder_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    steps: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    alg: str = "maskgit_plus",
    tokens_per_step: int | None = None,
):
    """Lightweight wrapper around DiffuCoder's diffusion_generate API."""
    attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
    outputs = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        alg=alg,
        alg_temp=0.0,
        tokens_per_step=tokens_per_step,
    )
    return outputs.sequences if hasattr(outputs, "sequences") else outputs
