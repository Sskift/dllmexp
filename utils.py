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
                generation = None
                if sample.get("resps"):
                    generation = sample["resps"][0][0]
                    if not isinstance(generation, str):
                        generation = str(generation)
                filtered = sample.get("filtered_resps", [None])[0]
                if filtered is not None and not isinstance(filtered, str):
                    filtered = str(filtered)

                # Fallback numeric extraction for GSM8K when post-processor yields [invalid]
                if task_name == "gsm8k" and (filtered in {None, "[invalid]", ""}):
                    extracted = _extract_last_number(generation or "")
                    if extracted is not None:
                        filtered = extracted

                parsed_samples.append(
                    {
                        "metric": metric_key,
                        "is_correct": bool(metric_value) if metric_value is not None else None,
                        "prompt": doc.get("question")
                        or doc.get("prompt")
                        or doc.get("text")
                        or doc.get("query")
                        or doc.get("instruction")
                        or doc.get("task_id"),
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

