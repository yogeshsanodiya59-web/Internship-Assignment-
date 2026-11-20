from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from eval.reference import TICKET_EXPECTATIONS, ExpectedOutcome
from question1.agent import SupportCopilot
from question2.agent import SupportPipeline
from shared.billing import CostSession

PUBLIC_TICKETS = ["T-1001", "T-1002", "T-1003", "T-1004"]
HOLDOUT_TICKETS = ["T-1091"]


@dataclass
class TicketBenchmarkResult:
    track: str
    ticket_id: str
    calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    points: int
    action: str
    expected_action: str
    kb_article: str
    expected_kb_article: str
    is_correct: bool
    target_met: bool


def benchmark_q1(tickets: list[str]) -> list[TicketBenchmarkResult]:
    results = []
    for tid in tickets:
        session = CostSession()
        copilot = SupportCopilot(session=session)
        res = copilot.handle_ticket(tid)
        expected = TICKET_EXPECTATIONS[tid]

        total_input = sum(c.input_tokens for c in session.calls)
        total_cached = sum(c.cached_prefix_tokens for c in session.calls)
        total_output = sum(c.output_tokens for c in session.calls)
        points = session.total_points()

        is_correct = (res.action == expected.action) and (res.kb_article == expected.kb_article)
        target_met = points <= 3000

        results.append(
            TicketBenchmarkResult(
                track="Q1 (Single Call)",
                ticket_id=tid,
                calls=len(session.calls),
                input_tokens=total_input,
                cached_input_tokens=total_cached,
                output_tokens=total_output,
                points=points,
                action=res.action,
                expected_action=expected.action,
                kb_article=res.kb_article,
                expected_kb_article=expected.kb_article,
                is_correct=is_correct,
                target_met=target_met,
            )
        )
    return results


def benchmark_q2(tickets: list[str]) -> list[TicketBenchmarkResult]:
    results = []
    for tid in tickets:
        session = CostSession()
        pipe = SupportPipeline(session=session)
        res = pipe.handle_ticket(tid)
        expected = TICKET_EXPECTATIONS[tid]

        total_input = sum(c.input_tokens for c in session.calls)
        total_cached = sum(c.cached_prefix_tokens for c in session.calls)
        total_output = sum(c.output_tokens for c in session.calls)
        points = session.total_points()

        is_correct = (res.action == expected.action) and (res.kb_article == expected.kb_article)
        target_met = points <= 2500

        results.append(
            TicketBenchmarkResult(
                track="Q2 (Pipeline)",
                ticket_id=tid,
                calls=len(session.calls),
                input_tokens=total_input,
                cached_input_tokens=total_cached,
                output_tokens=total_output,
                points=points,
                action=res.action,
                expected_action=expected.action,
                kb_article=res.kb_article,
                expected_kb_article=expected.kb_article,
                is_correct=is_correct,
                target_met=target_met,
            )
        )
    return results


def print_table(results: list[TicketBenchmarkResult]) -> None:
    headers = [
        "Track",
        "Ticket",
        "Calls",
        "Input",
        "Cached",
        "Output",
        "Points",
        "Action (Exp)",
        "KB Article (Exp)",
        "Correct",
        "Target Met",
    ]
    row_fmt = "{:<18} {:<8} {:<6} {:<8} {:<8} {:<8} {:<8} {:<24} {:<30} {:<8} {:<10}"
    print("-" * 144)
    print(row_fmt.format(*headers))
    print("-" * 144)
    for r in results:
        act_str = f"{r.action} ({r.expected_action})"
        kb_str = f"{r.kb_article} ({r.expected_kb_article})"
        correct_str = "PASS" if r.is_correct else "FAIL"
        target_str = "PASS" if r.target_met else "FAIL"
        print(
            row_fmt.format(
                r.track,
                r.ticket_id,
                r.calls,
                r.input_tokens,
                r.cached_input_tokens,
                r.output_tokens,
                r.points,
                act_str[:23],
                kb_str[:29],
                correct_str,
                target_str,
            )
        )
    print("-" * 144)


def main() -> None:
    print("=" * 60)
    print("SHOPASSIST COST OPTIMIZATION BENCHMARK")
    print("=" * 60)

    print("\n--- Question 1: Single Call Support Copilot ---")
    q1_results = benchmark_q1(PUBLIC_TICKETS + HOLDOUT_TICKETS)
    print_table(q1_results)

    print("\n--- Question 2: Four-Stage Support Pipeline ---")
    q2_results = benchmark_q2(PUBLIC_TICKETS)
    print_table(q2_results)


if __name__ == "__main__":
    main()

