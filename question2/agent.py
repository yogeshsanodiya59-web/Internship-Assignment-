from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from shared.billing import CostSession
from shared.kb import article_ids, expanded_help_center_blob, load_article, title_index
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import ALLOWED_ACTIONS, parse_resolution


@dataclass
class TicketResolution:
    action: str
    kb_article: str


class SupportPipeline:
    """
    ShopAssist Tier-1 Support Pipeline.

    Four-stage modular architecture: Triage -> Retrieve -> Compose -> Audit.
    Supports both cost-optimized mode (default) and original baseline mode.
    """

    def __init__(self, session: CostSession | None = None, baseline: bool = False):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.baseline = baseline
        self._help_center_cache: str | None = None

    @property
    def help_center(self) -> str:
        if self._help_center_cache is None:
            self._help_center_cache = expanded_help_center_blob()
        return self._help_center_cache

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)

        if self.baseline:
            snapshot = json.dumps(order, indent=2)
            triage = self._baseline_triage(ticket_id, thread, snapshot)
            retrieved = self._baseline_retrieve(triage, snapshot)
            draft = self._baseline_compose(ticket_id, thread, snapshot, triage, retrieved)
            final = self._baseline_audit(ticket_id, thread, snapshot, triage, retrieved, draft)
            parsed = parse_resolution(final)
            return TicketResolution(parsed.action, parsed.kb_article)

        # Cost-optimized mode
        snapshot = json.dumps(order, separators=(",", ":"))
        triage = self._triage(ticket_id, thread, snapshot)
        retrieved = self._retrieve(triage, snapshot)
        draft = self._compose(ticket_id, thread, snapshot, triage, retrieved)
        final = self._audit(ticket_id, thread, snapshot, triage, retrieved, draft)
        parsed = parse_resolution(final)
        return TicketResolution(parsed.action, parsed.kb_article)

    # ---------------------------------------------------------
    # Optimized Pipeline Nodes
    # ---------------------------------------------------------

    def _triage(self, ticket_id: str, thread: str, snapshot: str) -> str:
        """
        Stage 1: Triage Node (Deterministic Python Extraction).
        Extracts customer intent, issue category, and key order facts.
        """
        order_data = json.loads(snapshot) if snapshot else {}
        thread_lower = thread.lower()

        intent = "general_inquiry"
        if "refund" in thread_lower or "cancel" in thread_lower:
            intent = "refund_request"
        elif "replace" in thread_lower or "broken" in thread_lower or "damage" in thread_lower:
            intent = "replacement_request"

        triage_summary = {
            "ticket_id": ticket_id,
            "intent": intent,
            "product_type": order_data.get("product_type", "physical"),
            "damage_reported": bool(order_data.get("damage_reported")),
            "customized": bool(order_data.get("customized")),
            "license_revealed": bool(order_data.get("license_revealed")),
            "opened": bool(order_data.get("opened")),
            "delivered_days_ago": order_data.get("delivered_days_ago"),
            "subscription_age_days": order_data.get("subscription_age_days"),
        }
        return json.dumps(triage_summary, separators=(",", ":"))

    def _retrieve(self, triage: str, snapshot: str) -> str:
        """
        Stage 2: Retrieve Node (Targeted KB Search).
        Retrieves only the candidate policy articles relevant to the triaged intent.
        """
        triage_data = json.loads(triage) if triage else {}
        ptype = triage_data.get("product_type", "physical")

        candidate_ids: list[str] = []

        if ptype == "subscription" or triage_data.get("subscription_age_days") is not None:
            candidate_ids.append("subscription-cooling-off")

        if triage_data.get("license_revealed"):
            candidate_ids.append("digital-license-no-refund")

        if triage_data.get("customized"):
            candidate_ids.append("custom-items-no-replace")

        if triage_data.get("damage_reported"):
            candidate_ids.append("damage-replacement-7-days")

        if ptype == "physical" or not candidate_ids:
            candidate_ids.append("returns-within-30-days")

        seen = set()
        unique_ids = [aid for aid in candidate_ids if not (aid in seen or seen.add(aid))]

        snippets = []
        for aid in unique_ids:
            try:
                snippets.append(f"[{aid}]\n{load_article(aid).strip()}")
            except Exception:
                continue
        return "\n\n".join(snippets)

    def _compose(
        self,
        ticket_id: str,
        thread: str,
        snapshot: str,
        triage: str,
        retrieved: str,
    ) -> str:
        """
        Stage 3: Compose Node (Single LLM Call).
        Uses a concise, direct prompt without verbose trigger phrases to synthesize
        the ticket resolution.
        """
        messages = [
            ChatMessage(
                "system",
                "You are ShopAssist Compose. "
                "Resolve the ticket using the retrieved policies and order snapshot. "
                "Respond ONLY with a JSON object containing keys: action, kb_article, customer_reply. "
                "Valid actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2.",
            ),
            ChatMessage(
                "user",
                f"TICKET_ID: {ticket_id}\n"
                f"THREAD:\n{thread}\n\n"
                f"RETRIEVED_POLICIES:\n{retrieved}\n\n"
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return self.llm.chat("compose", messages)

    def _audit(
        self,
        ticket_id: str,
        thread: str,
        snapshot: str,
        triage: str,
        retrieved: str,
        draft: str,
    ) -> str:
        """
        Stage 4: Audit Node (Deterministic Guardrail).
        Validates the generated JSON schema, checks action validity, verifies article existence,
        and enforces policy rules deterministically without an unnecessary LLM call.
        """
        try:
            m = re.search(r"\{[\s\S]*\}", draft)
            if not m:
                raise ValueError("No JSON found in draft")
            data = json.loads(m.group())
        except Exception:
            data = {"action": "ESCALATE_TIER2", "kb_article": "returns-within-30-days", "customer_reply": ""}

        action = str(data.get("action", "")).upper()
        kb_article = str(data.get("kb_article", ""))

        if action not in ALLOWED_ACTIONS:
            action = "ESCALATE_TIER2"

        valid_ids = set(article_ids())
        if kb_article not in valid_ids:
            kb_article = "returns-within-30-days"

        # Policy consistency check
        triage_data = json.loads(triage) if triage else {}
        if triage_data.get("customized") and triage_data.get("damage_reported"):
            action = "ESCALATE_TIER2"
            kb_article = "custom-items-no-replace"
        elif triage_data.get("license_revealed"):
            action = "SEND_ARTICLE"
            kb_article = "digital-license-no-refund"

        audited_data = {
            "action": action,
            "kb_article": kb_article,
            "customer_reply": data.get("customer_reply", ""),
        }
        return json.dumps(audited_data, indent=2)

    # ---------------------------------------------------------
    # Baseline Implementation (Preserved for Benchmark & Testing)
    # ---------------------------------------------------------

    def _baseline_triage(self, ticket_id: str, thread: str, snapshot: str) -> str:
        messages = [
            ChatMessage(
                "system",
                "\n".join(
                    [
                        f"trace-{uuid.uuid4()}",
                        "Triage node: restate the ticket step by step and list candidate kb articles.",
                        f"ARTICLE TITLES:\n{title_index(500)}",
                        self.help_center[:120_000],
                    ]
                ),
            ),
            ChatMessage(
                "user",
                f"TICKET {ticket_id}\nTHREAD:\n{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return self.llm.chat("triage", messages)

    def _baseline_retrieve(self, triage: str, snapshot: str) -> str:
        messages = [
            ChatMessage(
                "system",
                "Retrieve node: print full text of the eight most relevant help articles.",
            ),
            ChatMessage(
                "user",
                f"TRIAGE:\n{triage}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return self.llm.chat("retrieve", messages)

    def _baseline_compose(
        self,
        ticket_id: str,
        thread: str,
        snapshot: str,
        triage: str,
        retrieved: str,
    ) -> str:
        messages = [
            ChatMessage(
                "system",
                "Compose node: full reasoning, then JSON with action, kb_article, customer_reply.",
            ),
            ChatMessage(
                "user",
                "\n\n".join(
                    [
                        f"TICKET {ticket_id}",
                        f"THREAD:\n{thread}",
                        f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
                        f"TRIAGE:\n{triage}",
                        f"ARTICLES:\n{retrieved}",
                        f"TRIAGE AGAIN:\n{triage}",
                    ]
                ),
            ),
        ]
        return self.llm.chat("compose", messages)

    def _baseline_audit(
        self,
        ticket_id: str,
        thread: str,
        snapshot: str,
        triage: str,
        retrieved: str,
        draft: str,
    ) -> str:
        messages = [
            ChatMessage(
                "system",
                "Audit node: double-check policy, restate ticket, rewrite JSON output.",
            ),
            ChatMessage(
                "user",
                "\n\n".join(
                    [
                        self.help_center[:80_000],
                        f"TICKET {ticket_id}",
                        f"THREAD:\n{thread}",
                        f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
                        f"DRAFT:\n{draft}",
                        f"TRIAGE:\n{triage}",
                        f"ARTICLES:\n{retrieved}",
                    ]
                ),
            ),
        ]
        return self.llm.chat("audit", messages)
