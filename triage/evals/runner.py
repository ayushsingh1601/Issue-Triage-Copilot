"""Comparison runner: vanilla RAG vs multi-agent over held-out issues,
plus a doc-chunking sweep. All LLM components are injectable for tests."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from triage.agents.historical import HistoricalAgent
from triage.agents.historical import default_model as historical_default_model
from triage.agents.process import ProcessAgent
from triage.agents.process import default_model as process_default_model
from triage.agents.vanilla import VanillaAgent, default_respond_fn
from triage.evals.judge import Judge
from triage.evals.metrics import action_overlap, label_accuracy, recall_at_k
from triage.guardrails.schema import TriageDecision
from triage.mcp_tools.langchain import AgentToolbox
from triage.mcp_tools.tools import TriageTools
from triage.orchestration.graph import build_graph
from triage.orchestration.nodes import default_orchestrator_respond
from triage.orchestration.state import TriageState
from triage.persist import load_records
from triage.prompts.judge import METRICS
from triage.rag.embed import Embedder
from triage.rag.index_build import build_doc_index, build_issue_index
from triage.rag.parse import IssueRecord, ProcessDoc
from triage.rag.pipeline import VanillaPipeline
from triage.rag.retriever import Retriever
from triage.rag.store import Match

COST_PER_CALL = 0.0002


@dataclass
class Components:
    tools: TriageTools | None = None
    vanilla_respond: Callable[[str], str] | None = None
    orchestrator_respond: Callable[[str], str] | None = None
    historical_model: Any | None = None
    process_model: Any | None = None
    judge_respond: Any | None = None


@dataclass
class SystemResults:
    system: str
    label_top1: float = 0.0
    label_top3: float = 0.0
    action_rouge_l: float = 0.0
    action_entity_match: float = 0.0
    answer_relevancy: float = 0.0
    context_relevance: float = 0.0
    groundedness: float = 0.0
    recall_at_k: float = 0.0
    latency_p95: float = 0.0
    cost: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {key: value for key, value in asdict(self).items() if key != "system"}


class CountingRespond:
    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn
        self.count = 0

    def __call__(self, prompt: str) -> str:
        self.count += 1
        return self._fn(prompt)


class CountingModel:
    def __init__(self, model: Any) -> None:
        self._model = model
        self.count = 0

    async def ainvoke(self, messages) -> Any:
        self.count += 1
        return await self._model.ainvoke(messages)


class EvaluationRunner:
    def __init__(
        self,
        indexes_dir: Path,
        processed_dir: Path,
        components: Components | None = None,
    ) -> None:
        comp = components or Components()
        self._tools = comp.tools or TriageTools(indexes_dir, processed_dir)
        self._corpus = load_records(processed_dir / "issues_corpus.json", IssueRecord)
        self._held_out = load_records(processed_dir / "issues_held_out.json", IssueRecord)
        self._retriever = self._tools.retriever()

        self._vanilla_respond = CountingRespond(comp.vanilla_respond or default_respond_fn())
        self._orchestrator_respond = CountingRespond(
            comp.orchestrator_respond or default_orchestrator_respond()
        )
        self._historical_model = CountingModel(comp.historical_model or historical_default_model())
        self._process_model = CountingModel(comp.process_model or process_default_model())

        self._vanilla_agent = VanillaAgent(self._vanilla_respond)
        self._historical_agent = HistoricalAgent(self._historical_model)
        self._process_agent = ProcessAgent(self._process_model)
        self._judge = Judge(comp.judge_respond)

    def run_comparison(self) -> dict[str, SystemResults]:
        return {
            "vanilla": self.evaluate_system("vanilla"),
            "multi": self.evaluate_system("multi"),
        }

    def evaluate_system(self, system: str) -> SystemResults:
        metric_scores: dict[str, list[float]] = {
            "label_top1": [],
            "label_top3": [],
            "action_rouge_l": [],
            "action_entity_match": [],
            "recall": [],
        }
        for metric in METRICS:
            metric_scores[metric] = []
        latencies: list[float] = []

        for record in self._held_out:
            query = f"{record.title}\n\n{record.body}"
            issue_id = f"{record.repo}#{record.number}"
            start = time.perf_counter()
            if system == "vanilla":
                decision = self._run_vanilla(query, issue_id)
            else:
                decision = asyncio.run(self._run_multi(query, issue_id))
            latencies.append(time.perf_counter() - start)

            label = label_accuracy(decision.suggested_labels, record.labels)
            metric_scores["label_top1"].append(label["top1"])
            metric_scores["label_top3"].append(label["top3"])

            reference = record.comments[-1].body if record.comments else ""
            overlap = action_overlap(decision.next_steps, reference, record.linked_prs)
            metric_scores["action_rouge_l"].append(overlap["rouge_l"])
            metric_scores["action_entity_match"].append(overlap["entity_match"])

            issue_matches, doc_matches = self._retriever.retrieve(
                query, k_issues=10, k_docs=3
            )
            relevant = self._relevant_ids(record)
            metric_scores["recall"].append(
                recall_at_k([match.id for match in issue_matches], relevant)
            )
            judged = self._judge.evaluate(
                query,
                decision.model_dump_json(),
                _context_text(issue_matches, doc_matches),
            )
            for metric in METRICS:
                metric_scores[metric].append(float(judged[metric].score))

        results = SystemResults(system=system)
        results.label_top1 = _mean(metric_scores["label_top1"])
        results.label_top3 = _mean(metric_scores["label_top3"])
        results.action_rouge_l = _mean(metric_scores["action_rouge_l"])
        results.action_entity_match = _mean(metric_scores["action_entity_match"])
        results.answer_relevancy = _mean(metric_scores["answer_relevancy"])
        results.context_relevance = _mean(metric_scores["context_relevance"])
        results.groundedness = _mean(metric_scores["groundedness"])
        results.recall_at_k = _mean(metric_scores["recall"])
        results.latency_p95 = _p95(latencies)
        results.cost = self._cost_per_triage(system, len(self._held_out))
        return results

    def _run_vanilla(self, query: str, issue_id: str) -> TriageDecision:
        return VanillaPipeline(retriever=self._retriever, agent=self._vanilla_agent).run(
            query, issue_id
        )

    async def _run_multi(self, query: str, issue_id: str) -> TriageDecision:
        async with AgentToolbox(self._tools) as box:
            graph = build_graph(
                toolbox=box,
                historical_agent=self._historical_agent,
                process_agent=self._process_agent,
                orchestrator_respond=self._orchestrator_respond,
            ).compile()
            result = await graph.ainvoke(TriageState(issue=query, issue_id=issue_id))
        assert result["decision"] is not None
        return result["decision"]

    def _relevant_ids(self, record: IssueRecord) -> set[str]:
        return {
            f"{item.repo}#{item.number}"
            for item in self._corpus
            if set(item.labels) & set(record.labels)
        }

    def _cost_per_triage(self, system: str, n: int) -> float:
        if n == 0:
            return 0.0
        if system == "vanilla":
            calls = self._vanilla_respond.count
        else:
            calls = n + self._historical_model.count + self._process_model.count
            calls += self._orchestrator_respond.count
        return round(calls * COST_PER_CALL / n, 6)


def sweep_doc_chunking(
    processed_dir: Path,
    indexes_base: Path,
    embedder: Embedder,
    judge_respond: Any | None = None,
    held_out_limit: int | None = None,
    configs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    docs = load_records(processed_dir / "process_docs.json", ProcessDoc)
    corpus = load_records(processed_dir / "issues_corpus.json", IssueRecord)
    held_out = load_records(processed_dir / "issues_held_out.json", IssueRecord)[
        :held_out_limit
    ]
    judge = Judge(judge_respond)
    configs = configs or [
        {"chunk_size": size, "overlap": overlap, "structural": structural}
        for structural in (True, False)
        for size in (300, 500, 700)
        for overlap in (0.1, 0.15)
    ]

    issue_store = build_issue_index(corpus, embedder, indexes_base)
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        doc_store = build_doc_index(
            docs,
            embedder,
            indexes_base / "sweep",
            collection=f"sweep_{index}",
            **config,
        )
        retriever = Retriever(
            embedder=embedder, issue_store=issue_store, doc_store=doc_store
        )
        recalls: list[float] = []
        relevance: list[float] = []
        for record in held_out:
            query = f"{record.title}\n\n{record.body}"
            relevant = {
                f"{item.repo}#{item.number}"
                for item in corpus
                if set(item.labels) & set(record.labels)
            }
            issue_matches, doc_matches = retriever.retrieve(
                query, k_issues=10, k_docs=3
            )
            recalls.append(
                recall_at_k([match.id for match in issue_matches], relevant)
            )
            score = judge.score(
                "context_relevance",
                query,
                "",
                _context_text(issue_matches, doc_matches),
            )
            relevance.append(float(score.score))
        rows.append(
            {
                **config,
                "recall_at_10": round(_mean(recalls), 4),
                "context_relevance": round(_mean(relevance), 4),
            }
        )
    return rows


def _context_text(issue_matches: list[Match], doc_matches: list[Match]) -> str:
    issues = "\n".join(match.text for match in issue_matches[:3])
    docs = "\n".join(match.text for match in doc_matches)
    return "\n".join(part for part in (issues, docs) if part)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return round(ordered[index], 6)
