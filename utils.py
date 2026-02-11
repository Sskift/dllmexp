import numpy as np


def _to_serializable(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def parse_results(results, task_name):
    output_data = {
        "metrics": {},
        "profile": results.get("profile", {}),
        "samples": {},
    }

    if "results" in results:
        for task, task_results in results["results"].items():
            output_data["metrics"][task] = {metric: _to_serializable(value) for metric, value in task_results.items()}

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
                filtered = sample.get("filtered_resps", [None])[0]

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
                        "target": doc.get("answer")
                        or doc.get("canonical_solution")
                        or doc.get("label")
                        or doc.get("target")
                        or doc.get("test")
                        or doc.get("tests"),
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

