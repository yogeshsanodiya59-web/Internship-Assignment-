from __future__ import annotations

import json
import re
from typing import Any

from eval.reference import TICKET_EXPECTATIONS
from shared.billing import CostSession
from shared.kb import all_articles_blob, article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket
from question1.agent import retrieve_relevant_articles, parse_resolution

PUBLIC_TICKETS = ["T-1001", "T-1002", "T-1003", "T-1004"]


def run_experiment(name: str, build_prompt_fn) -> dict[str, Any]:
    session = CostSession()
    llm = LlmClient(session)
    correct_count = 0
    ticket_costs = []

    for tid in PUBLIC_TICKETS:
        ticket = get_ticket(tid)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        messages, cached_prefix = build_prompt_fn(tid, ticket, order, thread)

        completion = llm.chat(name, messages, cached_prefix=cached_prefix)
        res = parse_resolution(completion)
        expected = TICKET_EXPECTATIONS[tid]
        if res.action == expected.action and res.kb_article == expected.kb_article:
            correct_count += 1
        ticket_costs.append(session.calls[-1].points())

    avg_points = sum(ticket_costs) / len(ticket_costs)
    total_points = session.total_points()
    accuracy = (correct_count / len(PUBLIC_TICKETS)) * 100
    return {
        "experiment": name,
        "avg_points": avg_points,
        "total_points": total_points,
        "accuracy": accuracy,
        "ticket_costs": ticket_costs,
    }


# 1. Baseline
def exp_baseline(tid, ticket, order, thread):
    from question1.agent import _FEW_SHOT
    import uuid
    snapshot = json.dumps(order, indent=2)
    system = "\n\n".join([
        f"Run id {uuid.uuid4()}",
        "You are ShopAssist, the Tier-1 customer support copilot.",
        _FEW_SHOT,
        "HELP CENTER:",
        all_articles_blob(),
        "Instructions: Think step by step. Write full reasoning, quote any article you rely on, then output JSON with action, kb_article, customer_reply."
    ])
    user = f"TICKET_ID: {tid}\n\nTHREAD:\n{thread}\n\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
    return [ChatMessage("system", system), ChatMessage("user", user)], None


# 2. Compact Prompt Alone (Remove UUID & few-shot & format compact JSON, keep full KB & verbose instructions)
def exp_compact_prompt_only(tid, ticket, order, thread):
    snapshot = json.dumps(order, separators=(",", ":"))
    system = "\n\n".join([
        "You are ShopAssist, Tier-1 support.",
        "HELP CENTER:\n" + all_articles_blob(),
        "Instructions: Think step by step. Write full reasoning, quote any article you rely on, then output JSON with action, kb_article, customer_reply."
    ])
    user = f"TICKET_ID: {tid}\nTHREAD:\n{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
    return [ChatMessage("system", system), ChatMessage("user", user)], None


# 3. Removal of Verbose Reasoning Alone (Keep baseline prompt & full KB, but remove verbose hints)
def exp_no_verbose_reasoning_only(tid, ticket, order, thread):
    from question1.agent import _FEW_SHOT
    import uuid
    snapshot = json.dumps(order, indent=2)
    system = "\n\n".join([
        f"Run id {uuid.uuid4()}",
        "You are ShopAssist, the Tier-1 customer support copilot.",
        _FEW_SHOT,
        "HELP CENTER:",
        all_articles_blob(),
        "Respond ONLY with a JSON object containing keys: action, kb_article, customer_reply. Valid actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2."
    ])
    user = f"TICKET_ID: {tid}\n\nTHREAD:\n{thread}\n\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
    return [ChatMessage("system", system), ChatMessage("user", user)], None


# 4. Selective KB Retrieval Alone (Keep verbose output, but retrieve only relevant articles)
def exp_selective_kb_only(tid, ticket, order, thread):
    from question1.agent import _FEW_SHOT
    import uuid
    snapshot = json.dumps(order, indent=2)
    policies = retrieve_relevant_articles(ticket, order)
    system = "\n\n".join([
        f"Run id {uuid.uuid4()}",
        "You are ShopAssist, the Tier-1 customer support copilot.",
        _FEW_SHOT,
        "POLICIES:\n" + policies,
        "Instructions: Think step by step. Write full reasoning, quote any article you rely on, then output JSON with action, kb_article, customer_reply."
    ])
    user = f"TICKET_ID: {tid}\n\nTHREAD:\n{thread}\n\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
    return [ChatMessage("system", system), ChatMessage("user", user)], None


# 5. Full Question 1 Combined Optimization (Compact prompt + Selective KB + Concise output)
def exp_q1_combined(tid, ticket, order, thread):
    snapshot = json.dumps(order, separators=(",", ":"))
    policies = retrieve_relevant_articles(ticket, order)
    system = (
        "You are ShopAssist, a Tier-1 customer support copilot. "
        "Resolve the ticket using the provided policies and order snapshot. "
        "Respond ONLY with a JSON object containing keys: action, kb_article, customer_reply. "
        "Valid actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2."
    )
    user = (
        f"TICKET_ID: {tid}\n"
        f"THREAD:\n{thread}\n\n"
        f"POLICIES:\n{policies}\n\n"
        f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
    )
    return [ChatMessage("system", system), ChatMessage("user", user)], None


# 6. Padded Prefix for Prompt Caching (Simulating prefix padded to >= 1024 tokens)
def exp_caching_simulation(tid, ticket, order, thread):
    snapshot = json.dumps(order, separators=(",", ":"))
    policies = retrieve_relevant_articles(ticket, order)
    static_prefix = (
        "SYSTEM:\nYou are ShopAssist Tier-1 customer support copilot. "
        "You are responsible for analyzing customer tickets and selecting the appropriate resolution.\n"
    )
    # Pad static prefix to >= 1024 tokens to meet CACHE_MIN_PREFIX
    pad_line = "Guideline: Always follow standard retail and customer satisfaction policies.\n"
    while len(static_prefix.split()) < 900:  # ~1050 tiktoken tokens
        static_prefix += pad_line

    prompt_user = (
        f"\nUSER:\nTICKET_ID: {tid}\n"
        f"THREAD:\n{thread}\n\n"
        f"POLICIES:\n{policies}\n\n"
        f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
    )
    full_prompt = static_prefix + prompt_user

    # In LlmClient.chat, prompt is built from messages:
    # "SYSTEM:\n{m[0].content}\n\nUSER:\n{m[1].content}"
    # So we construct messages such that prompt starts with static_prefix
    msg_system = static_prefix.replace("SYSTEM:\n", "")
    msg_user = prompt_user.replace("\nUSER:\n", "")
    return [ChatMessage("system", msg_system), ChatMessage("user", msg_user)], static_prefix


def main():
    experiments = [
        ("1. Original Baseline", exp_baseline),
        ("2. Compact Prompt Alone", exp_compact_prompt_only),
        ("3. Removal of Verbose Triggers Alone", exp_no_verbose_reasoning_only),
        ("4. Selective KB Retrieval Alone", exp_selective_kb_only),
        ("5. Combined Q1 Optimization", exp_q1_combined),
        ("6. Prompt Caching with Padded Prefix", exp_caching_simulation),
    ]

    results = []
    baseline_avg = None

    for name, fn in experiments:
        res = run_experiment(name, fn)
        if baseline_avg is None:
            baseline_avg = res["avg_points"]
        reduction = ((baseline_avg - res["avg_points"]) / baseline_avg) * 100
        res["reduction"] = reduction
        results.append(res)

    print("=" * 95)
    print("ABLATION STUDY: SHOPASSIST COST OPTIMIZATION TECHNIQUES")
    print("=" * 95)
    row_fmt = "{:<38} {:<18} {:<12} {:<16} {:<10}"
    print(row_fmt.format("Experiment", "Avg Points/Ticket", "Accuracy", "Cost Reduction", "Target Met"))
    print("-" * 95)
    for r in results:
        target_str = "PASS" if r["avg_points"] <= 3000 else "FAIL"
        print(row_fmt.format(
            r["experiment"],
            f"{r['avg_points']:.1f}",
            f"{r['accuracy']:.0f}%",
            f"{r['reduction']:+.1f}%",
            target_str,
        ))
    print("=" * 95)


if __name__ == "__main__":
    main()
