import torch
import numpy as np
import torch.nn.functional as F

def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    return scores_normalized < probs_thresh


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def llada_diffusion_generate(
    model,
    prompt,
    num_steps=128,
    gen_length=128,
    block_length=128,
    tokens_per_step=None,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert num_steps % num_blocks == 0
    steps = num_steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length] == mask_id
        )
        # If tokens_per_step is provided, use it directly (capped by remaining masks); otherwise keep uniform schedule
        num_transfer_tokens = None if tokens_per_step is not None else get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                remaining = (block_mask_index[j]).sum().item()
                k = min(tokens_per_step if tokens_per_step is not None else num_transfer_tokens[j, i].item(), remaining)
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def llada_sdola_diffusion_generate(
    model,
    prompt,
    num_steps=128,
    gen_length=128,
    block_length=128,
    tokens_per_step=None,
    temperature=0.0,
    mask_id=126336,
    mature_layer=None,
    premature_layer=None,
    relative_top=0.1,
    relative_top_value=-1000.0,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
        mature_layer: The mature layer index. Defaults to None (which means the last layer -1).
        premature_layer: The premature layer index.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert num_steps % num_blocks == 0
    steps = num_steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length] == mask_id
        )
        # If tokens_per_step is provided, use it directly (capped by remaining masks); otherwise keep uniform schedule
        num_transfer_tokens = None if tokens_per_step is not None else get_num_transfer_tokens(block_mask_index, steps)
        
        # Determine layers once per generation to avoid recalculating
        if mature_layer is None:
            mature_layer = -1
            
        if premature_layer is None:
            # We need to do a dummy forward pass to get the number of layers
            dummy_outputs = model(x, output_hidden_states=True)
            num_layers = len(dummy_outputs.hidden_states) - 1
            premature_layer = num_layers // 2
            
        for i in range(steps):
            mask_index = x == mask_id
            
            outputs = model(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            mature_hidden = hidden_states[mature_layer]
            premature_hidden = hidden_states[premature_layer]
            
            mature_logits = model.model.transformer.ff_out(model.model.transformer.ln_f(mature_hidden))
            premature_logits = model.model.transformer.ff_out(model.model.transformer.ln_f(premature_hidden))
            
            final_logits = mature_logits.log_softmax(dim=-1)
            base_logits = premature_logits.log_softmax(dim=-1)
            diff_logits = final_logits - base_logits
            
            if relative_top > 0.0:
                relative_top_mask = get_relative_top_filter(mature_logits, relative_top)
                diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
            
            logits = diff_logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                remaining = (block_mask_index[j]).sum().item()
                k = min(tokens_per_step if tokens_per_step is not None else num_transfer_tokens[j, i].item(), remaining)
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def llada_ddola_diffusion_generate(
    model,
    prompt,
    num_steps=128,
    gen_length=128,
    block_length=128,
    tokens_per_step=None,
    temperature=0.0,
    mask_id=126336,
    mature_layer=None,
    candidate_premature_layers=None,
    relative_top=0.1,
    relative_top_value=-1000.0,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
        mature_layer: The mature layer index. Defaults to None (which means the last layer -1).
        candidate_premature_layers: List of candidate premature layers.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert num_steps % num_blocks == 0
    steps = num_steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length] == mask_id
        )
        # If tokens_per_step is provided, use it directly (capped by remaining masks); otherwise keep uniform schedule
        num_transfer_tokens = None if tokens_per_step is not None else get_num_transfer_tokens(block_mask_index, steps)
        
        # Determine layers once per generation to avoid recalculating
        if mature_layer is None:
            mature_layer = -1
            
        if candidate_premature_layers is None:
            # We need to do a dummy forward pass to get the number of layers
            dummy_outputs = model(x, output_hidden_states=True)
            num_layers = len(dummy_outputs.hidden_states) - 1
            start_layer = num_layers // 4
            end_layer = (num_layers * 3) // 4
            candidate_premature_layers = list(range(start_layer, end_layer))
            
        for i in range(steps):
            mask_index = x == mask_id
            
            outputs = model(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            mature_hidden = hidden_states[mature_layer]
            mature_logits = model.model.transformer.ff_out(model.model.transformer.ln_f(mature_hidden))
            
            # Calculate logits for all candidate premature layers
            candidate_logits = []
            for layer in candidate_premature_layers:
                premature_hidden = hidden_states[layer]
                layer_logits = model.model.transformer.ff_out(model.model.transformer.ln_f(premature_hidden))
                candidate_logits.append(layer_logits)
            
            # Stack candidate logits: (num_candidates, batch_size, seq_len, vocab_size)
            stacked_candidate_logits = torch.stack(candidate_logits, dim=0)
            
            # Calculate softmax distributions
            softmax_mature = F.softmax(mature_logits, dim=-1)
            softmax_candidates = F.softmax(stacked_candidate_logits, dim=-1)
            
            # Calculate M (average distribution)
            M = 0.5 * (softmax_mature.unsqueeze(0) + softmax_candidates)
            
            # Calculate log-softmax for KL divergence
            log_softmax_mature = F.log_softmax(mature_logits, dim=-1)
            log_softmax_candidates = F.log_softmax(stacked_candidate_logits, dim=-1)
            
            # Calculate KL divergences
            kl1 = F.kl_div(log_softmax_mature.unsqueeze(0), M, reduction='none').mean(-1) # (num_candidates, batch_size, seq_len)
            kl2 = F.kl_div(log_softmax_candidates, M, reduction='none').mean(-1) # (num_candidates, batch_size, seq_len)
            
            # Calculate JS divergences
            js_divs = 0.5 * (kl1 + kl2) # (num_candidates, batch_size, seq_len)
            
            # Find the layer with maximum JS divergence for each token
            max_js_indices = js_divs.argmax(dim=0)
            
            # Gather the base logits using the selected indices
            gather_indices = max_js_indices.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, mature_logits.size(-1))
            selected_base_logits = torch.gather(stacked_candidate_logits, 0, gather_indices).squeeze(0)
            
            final_logits = mature_logits.log_softmax(dim=-1)
            base_logits = selected_base_logits.log_softmax(dim=-1)
            diff_logits = final_logits - base_logits
            
            if relative_top > 0.0:
                relative_top_mask = get_relative_top_filter(mature_logits, relative_top)
                diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
            
            logits = diff_logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                remaining = (block_mask_index[j]).sum().item()
                k = min(tokens_per_step if tokens_per_step is not None else num_transfer_tokens[j, i].item(), remaining)
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def llada_ar_generate(model, prompt, num_steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, tokens_per_step=1):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert num_steps % num_blocks == 0
    steps = num_steps // num_blocks

    for i in range(steps):
        mask_index = (x == mask_id)
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

        mask = mask_index.clone()
        cumsum_mask = torch.cumsum(mask, dim=1)
        transfer_index = torch.logical_and(cumsum_mask >= 1, cumsum_mask <= tokens_per_step) # Gets first tokens_per_step
        
        x[transfer_index] = x0[transfer_index]
        
        no_mask = (x == mask_id).sum().item() == 0
        has_eos = (x == 126081).sum().item() > 0
        
        if no_mask or has_eos:
            break
            
    return x
