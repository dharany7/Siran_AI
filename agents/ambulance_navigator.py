"""
agents/ambulance_navigator.py — Gemini-powered route planner for emergency vehicles.

Given a start/destination junction and the live SUMO traffic state, calls Google
Gemini with a strict system prompt and parses its response as JSON:
    {"route": ["1","J1","J2","J3","3"], "reasoning": "...one paragraph..."}

The full reasoning text is logged at INFO level so it is visible in the demo
server console and captured in the negotiation log returned by POST /dispatch.

SDK: google-genai  (from google import genai)
Client pattern: client = genai.Client(api_key=...)
               response = client.models.generate_content(model=..., contents=...)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from agents.base_agent import BaseAgent
from backend.config import get_gemini_client, get_settings

logger = logging.getLogger(__name__)

# ── Network description (dynamically updated when sim starts; static fallback) ──
_NETWORK_DESCRIPTION = """\
Kilpauk, Chennai — Real OSM Road Network
=========================================
This is the real road network for Kilpauk, a medical district in Chennai, India.
Converted from OpenStreetMap data (bbox: N=13.097 S=13.070 E=80.253 W=80.227).

Network statistics:
  - ~2,275 real road junctions
  - ~5,202 directed road edges (SUMO edge IDs are OSM way IDs, e.g. "184720014")
  - 15 real traffic-signal-controlled junctions (TLS)

Key hospitals in this network:
  - Kilpauk Medical College & Hospital (13.0843, 80.2371)
  - KMCH Kauvery Medical Centre (13.0837, 80.2397)
  - Apollo Spectra Hospital (13.0856, 80.2415)
  - Vijaya Hospital (13.0770, 80.2340)

Routing:
  Routes are computed by SUMO duarouter (shortest-path through the real network).
  Each route is a list of SUMO edge IDs (OSM-derived, e.g. ["184720014", "186080195", ...]).
  You do NOT invent route edges — the duarouter has already computed the optimal path.
  Your job is to REASON about why this route is best given current traffic state,
  and to confirm or slightly adjust it based on live TLS states.
"""

_SYSTEM_PROMPT = f"""\
You are an emergency-vehicle routing AI integrated into a real traffic simulator.
Your sole task: reason about the fastest ambulance route and return structured JSON.

{_NETWORK_DESCRIPTION}

You will be given:
  - The pre-computed duarouter route (a list of SUMO edge IDs)
  - The current live traffic state (TLS phases, vehicle count)
  - Start and destination descriptions

OUTPUT RULES:
1. Your entire response MUST be valid JSON. No markdown. No text outside JSON.
2. Schema (exactly these keys, no others):
   {{"route": ["edge_id", ...], "reasoning": "one concise paragraph"}}
3. "route" MUST be the edge ID list you were given (confirm it or return it unchanged).
4. "reasoning" MUST mention: TLS junction count, vehicle density, estimated travel time,
   and why this path through Kilpauk's road network is optimal.
5. Do NOT invent new edge IDs — only use the ones provided to you.
"""


class AmbulanceNavigatorAgent(BaseAgent):
    """
    LLM agent that asks Gemini to reason about the fastest ambulance route.

    For the 3-TLS linear chain network, the only complete route from "1" to "3"
    is ["1","J1","J2","J3","3"] — Gemini reasons about which TLS junctions need
    preemption based on current traffic state.

    Payload keys
    ------------
    start       : str   — starting junction ID  (e.g. "1")
    destination : str   — destination junction ID (e.g. "3")
    sim_state   : dict  — current state from SumoEnv.get_state()

    Returns
    -------
    dict with:
        route        : list[str]  — ordered junction IDs
        reasoning    : str        — Gemini's reasoning paragraph
        raw_response : str        — exact model output (for demo log)
        model        : str        — Gemini model name used
        error        : str        — present only on failure (fallback route used)
    """

    def __init__(self) -> None:
        super().__init__(name="ambulance_navigator")

    # ── Public sync entry point (used directly by the sync dispatch handler) ──

    def run_sync(self, payload: dict) -> dict:
        start_desc  = str(payload.get("start", "start"))
        dest_desc   = str(payload.get("destination", "hospital"))
        sim_state   = payload.get("sim_state", {})
        # Pre-computed route from duarouter (list of edge IDs)
        precomputed_route: list[str] = payload.get("route", [])
        model       = get_settings().gemini_model

        route_preview = (
            f"['{precomputed_route[0]}', ..., '{precomputed_route[-1]}'] "
            f"({len(precomputed_route)} edges)"
            if len(precomputed_route) >= 2 else str(precomputed_route)
        )

        user_prompt = (
            f"An ambulance must travel from {start_desc!r} to {dest_desc!r}.\n\n"
            f"Pre-computed duarouter route ({len(precomputed_route)} edges):\n"
            f"  {route_preview}\n\n"
            f"Current live traffic state:\n"
            f"  TLS junctions in network    : {sim_state.get('tls_count', '?')}\n"
            f"  Sample TLS state            : {sim_state.get('tls_state', 'unknown')!r}\n"
            f"  Total vehicles in network   : {sim_state.get('vehicle_count', '?')}\n"
            f"  Ambulance already present   : {sim_state.get('ambulance_present', False)}\n\n"
            f"Return the route JSON confirming or refining the pre-computed route."
        )

        self.logger.info(
            "Calling Gemini (model=%s) for route %s -> %s", model, start_desc, dest_desc
        )
        self.logger.debug("User prompt:\n%s", user_prompt)

        raw_text = ""
        try:
            client   = get_gemini_client()
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature":        0.1,    # near-deterministic for JSON
                    "max_output_tokens":  1024,   # ≥512 to avoid cut-off reasoning
                },
            )
            raw_text = response.text.strip()
        except Exception as exc:
            self.logger.error("Gemini call failed: %s", exc)
            return {
                "route":        precomputed_route,
                "reasoning":    f"[FALLBACK] Gemini unavailable ({exc}). duarouter route used.",
                "raw_response": raw_text,
                "model":        model,
                "error":        str(exc),
            }

        # ── Log full response for demo visibility ─────────────────────────────
        self.logger.info(
            "=== GEMINI FULL RESPONSE ===\n%s\n=== END GEMINI RESPONSE ===", raw_text
        )

        # ── Parse and validate ────────────────────────────────────────────────
        try:
            parsed = self._parse_json(raw_text)
            if not isinstance(parsed, dict) or "reasoning" not in parsed:
                raise ValueError("Missing 'reasoning' key")
            reasoning = str(parsed["reasoning"]).strip()
            # Accept Gemini's route if it returned one, otherwise use the precomputed
            gemini_route = parsed.get("route", [])
            route = (
                [str(e) for e in gemini_route]
                if gemini_route and isinstance(gemini_route, list)
                else precomputed_route
            )
            # Fallback to precomputed if Gemini returned an empty or too-short route
            if len(route) < 2:
                route = precomputed_route
        except Exception as parse_exc:
            self.logger.warning("Gemini parse error: %s", parse_exc)
            reasoning = f"[unparsed] {raw_text[:300]}"
            route = precomputed_route

        self.logger.info("Navigator route: %d edges", len(route))
        self.logger.info("Navigator reasoning: %s", reasoning)

        return {
            "route":        route,
            "reasoning":    reasoning,
            "raw_response": raw_text,
            "model":        model,
        }

    # ── Async interface (satisfies BaseAgent ABC) ─────────────────────────────

    async def run(self, payload: dict) -> dict:
        return await asyncio.to_thread(self.run_sync, payload)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_fallback_route(self, start_edge: str, end_edge: str) -> list[str]:
        """
        Fallback: try duarouter; if unavailable just return [start, end].
        """
        try:
            from agents.grid_controller import route_via_duarouter
            return route_via_duarouter(start_edge, end_edge)
        except Exception as exc:
            self.logger.warning("duarouter fallback failed: %s", exc)
            return [start_edge, end_edge]

    def _parse_json(self, text: str) -> dict:
        """Strip markdown fences if any and parse JSON; fall back gracefully."""
        # Remove ```json ... ``` fences
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip("` \n")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to salvage the first {...} block
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        self.logger.warning("Could not parse Gemini JSON (%.120r) — using empty dict", text)
        return {}

    def _validate_route(self, route: list, start: str, dest: str) -> list[str]:
        """Accept any list of strings as edge IDs; validate non-empty."""
        route = [str(e) for e in route if str(e).strip()]
        if not route:
            route = self._build_fallback_route(start, dest)
        return route
