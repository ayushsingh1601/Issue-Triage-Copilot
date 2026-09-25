"""Outcome metrics: label accuracy, ROUGE-L action overlap, retrieval recall@k."""
from __future__ import annotations


def label_accuracy(predicted: list[str], actual: list[str]) -> dict[str, float]:
    actual_set = set(actual)
    return {
        "top1": 1.0 if predicted and predicted[0] in actual_set else 0.0,
        "top3": 1.0 if any(label in actual_set for label in predicted[:3]) else 0.0,
    }


def rouge_l_f1(predicted: str, reference: str) -> float:
    pred_tokens = predicted.split()
    ref_tokens = reference.split()
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0 or not pred_tokens or not ref_tokens:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def action_overlap(
    predicted_actions: list[str],
    reference: str,
    referenced_prs: list[int] | None = None,
) -> dict[str, float]:
    if not predicted_actions:
        return {"rouge_l": 0.0, "entity_match": 0.0}
    rouge = sum(rouge_l_f1(action, reference) for action in predicted_actions) / len(
        predicted_actions
    )
    joined = " ".join(predicted_actions)
    if referenced_prs:
        entity_match = sum(1 for pr in referenced_prs if str(pr) in joined) / len(referenced_prs)
    else:
        entity_match = 0.0
    return {"rouge_l": round(rouge, 4), "entity_match": round(entity_match, 4)}


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for issue_id in retrieved_ids if issue_id in relevant_ids)
    return hits / len(relevant_ids)


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
