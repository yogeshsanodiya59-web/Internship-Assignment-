from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from shared.billing import CostSession
from shared.kb import all_articles_blob, article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


ALLOWED_ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    action = str(data.get("action", "")).upper()
    if action not in ALLOWED_ACTIONS:
        action = "ESCALATE_TIER2"
    return TicketResolution(
        action=action,
        kb_article=str(data.get("kb_article", "")),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


_FEW_SHOT = """
Example ticket resolutions (format to copy):

1. Unopened headphones, day 10 → {"action":"REFUND","kb_article":"returns-within-30-days"}
2. Ebook, license shown → {"action":"SEND_ARTICLE","kb_article":"digital-license-no-refund"}
3. Cracked custom mug → {"action":"ESCALATE_TIER2","kb_article":"custom-items-no-replace"}
4. Cracked stock mug, day 2 → {"action":"REPLACE","kb_article":"damage-replacement-7-days"}
"""


def retrieve_relevant_articles(ticket: dict[str, Any], order: dict[str, Any]) -> str:
    """
    Selectively retrieve only relevant policy articles based on ticket content and order facts.
    Avoids dumping the entire knowledge base into the prompt.
    """
    thread_text = " ".join(m.get("text", "") for m in ticket.get("messages", [])).lower()
    ptype = str(order.get("product_type", "")).lower()

    selected_ids: list[str] = []

    # 1. Subscription policy
    if ptype == "subscription" or "subscription" in thread_text or "annual" in thread_text:
        selected_ids.append("subscription-cooling-off")

    # 2. Digital goods license policy
    if order.get("license_revealed") or "license" in thread_text or "digital" in thread_text or "ebook" in thread_text:
        selected_ids.append("digital-license-no-refund")

    # 3. Custom / personalized merchandise
    if order.get("customized") or "custom" in thread_text or "personalized" in thread_text or "engrav" in thread_text:
        selected_ids.append("custom-items-no-replace")

    # 4. Damaged physical goods
    if (
        order.get("damage_reported")
        or "damage" in thread_text
        or "broken" in thread_text
        or "cracked" in thread_text
        or "shatter" in thread_text
    ):
        selected_ids.append("damage-replacement-7-days")

    # 5. Standard returns
    if (
        ptype == "physical"
        or "return" in thread_text
        or "refund" in thread_text
        or not selected_ids
    ):
        selected_ids.append("returns-within-30-days")

    seen = set()
    unique_ids = [aid for aid in selected_ids if not (aid in seen or seen.add(aid))]

    snippets = []
    for aid in unique_ids:
        content = load_article(aid).strip()
        snippets.append(f"[{aid}]\n{content}")
    return "\n\n".join(snippets)


class SupportCopilot:
    """
    ShopAssist Tier-1 copilot.

    Single-call LLM agent supporting both cost-optimized mode (default)
    and original baseline mode (for benchmarking and regression testing).
    """

    def __init__(self, session: CostSession | None = None, baseline: bool = False):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.baseline = baseline

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)

        if self.baseline:
            # Original unoptimized baseline logic
            snapshot = json.dumps(order, indent=2)
            system = "\n\n".join(
                [
                    f"Run id {uuid.uuid4()}",
                    "You are ShopAssist, the Tier-1 customer support copilot.",
                    _FEW_SHOT,
                    "HELP CENTER:",
                    all_articles_blob(),
                    (
                        "Instructions: Think step by step. Write full reasoning, quote any article "
                        "you rely on, then output JSON with action, kb_article, customer_reply."
                    ),
                ]
            )
            user = "\n\n".join(
                [
                    f"TICKET_ID: {ticket_id}",
                    f"THREAD:\n{thread}",
                    f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
                ]
            )
            return [ChatMessage("system", system), ChatMessage("user", user)]

        # Cost-optimized single-call prompt
        snapshot = json.dumps(order, separators=(",", ":"))
        policy_context = retrieve_relevant_articles(ticket, order)

        system = (
            "You are ShopAssist, a Tier-1 customer support copilot. "
            "Resolve the ticket using the provided policies and order snapshot. "
            "Respond ONLY with a JSON object containing keys: action, kb_article, customer_reply. "
            "Valid actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2."
        )
        user = (
            f"TICKET_ID: {ticket_id}\n"
            f"THREAD:\n{thread}\n\n"
            f"POLICIES:\n{policy_context}\n\n"
            f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
        )
        return [ChatMessage("system", system), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)
