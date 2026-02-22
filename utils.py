import numpy as np
import re


def _to_serializable(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _extract_last_number(text: str):
    # Extract the last numeric token (integer or decimal) from text
    if not isinstance(text, str):
        return None
    # Drop currency symbols/commas for robustness
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return matches[-1] if matches else None


def _strip_prefix(text: str | None, prefix: str | None) -> str | None:
    if not text or not prefix:
        return text
    if text.startswith(prefix):
        return text[len(prefix):].lstrip()
    return text


def _normalize_resp(resp, prompt_text: str | None) -> str | None:
    """Normalize a response or filtered response to a plain string without the prompt prefix."""
    if resp is None:
        return None
    # If list/tuple, join and recurse
    if isinstance(resp, (list, tuple)):
        joined = " ".join(str(x) for x in resp)
        return _normalize_resp(joined, prompt_text)
    if not isinstance(resp, str):
        resp = str(resp)
    s = resp.strip()
    # If it is a repr of a single-element list like "['...']", unwrap it
    if s.startswith("['") and s.endswith("']"):
        s = s[2:-2].strip()
    # Remove one leading occurrence of the prompt
    s = _strip_prefix(s, prompt_text) or s
    return s


def _strip_suffix_tokens(text: str | None) -> str | None:
    """Strip common special-token suffixes such as <|im_end|>, <|eot_id|>, <|."""
    if text is None:
        return None
    suffixes = ["<|im_end|>", "<|im_end", "<|eot_id|>", "<|eot_id", "<|", "</s>"]
    trimmed = text
    for suf in suffixes:
        if suf in trimmed:
            trimmed = trimmed.split(suf, 1)[0]
    return trimmed.rstrip()


def parse_results(results, task_name):
    metric_filter = {
        "truthfulqa_gen": {"bleu_max,none", "rouge1_max,none", "rouge2_max,none", "rougeL_max,none"},
    }

    output_data = {
        "metrics": {},
        "profile": results.get("profile", {}),
        "samples": {},
    }

    if "results" in results:
        for task, task_results in results["results"].items():
            keep = metric_filter.get(task)
            filtered = {}
            for metric, value in task_results.items():
                if keep is None or metric in keep:
                    filtered[metric] = _to_serializable(value)
            output_data["metrics"][task] = filtered

    if "samples" in results:
        for task, sample_list in results["samples"].items():
            parsed_samples = []
            for sample in sample_list:
                metric_key = sample.get("metrics", [None])[0] if sample.get("metrics") else None
                metric_value = sample.get(metric_key) if metric_key else None
                doc = sample.get("doc", {}) or {}
                generation_raw = sample.get("resps", [None])[0]
                generation = _normalize_resp(generation_raw[0] if isinstance(generation_raw, (list, tuple)) else generation_raw, None)
                filtered_raw = sample.get("filtered_resps", [None])[0]

                prompt_text = doc.get("question") or doc.get("prompt") or doc.get("text") or doc.get("query") or doc.get("instruction") or doc.get("task_id")
                generation = _strip_suffix_tokens(_normalize_resp(generation, prompt_text))
                filtered = _strip_suffix_tokens(_normalize_resp(filtered_raw, prompt_text))

                # Fallback numeric extraction for GSM8K when post-processor yields [invalid]
                if task_name == "gsm8k" and (filtered in {None, "[invalid]", ""}):
                    extracted = _extract_last_number(generation or "")
                    if extracted is not None:
                        filtered = extracted

                parsed_samples.append(
                    {
                        "metric": metric_key,
                        "is_correct": bool(metric_value) if metric_value is not None else None,
                        "prompt": prompt_text,
                        "target": doc.get("best_answer")
                        or doc.get("correct_answers")
                        or doc.get("answer")
                        or doc.get("canonical_solution")
                        or doc.get("code")
                        or doc.get("label")
                        or doc.get("target")
                        or doc.get("test")
                        or doc.get("tests")
                        or doc.get("mc1_targets")
                        or doc.get("mc2_targets"),
                        "generation": generation.strip() if isinstance(generation, str) else generation,
                        "filtered": filtered,
                    }
                )

            output_data["samples"][task] = parsed_samples

    return output_data


def remove_masks(text):
    mask = "<|mask|>"
    while text.endswith(mask):
        text = text[: -len(mask)]
    return text

