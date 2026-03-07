#!/usr/bin/env python3
"""
Attacker Agent - Red team agent for security testing.

Receives scenario context as JSON, builds strategic prompts, and generates attacks.
Works across different scenarios by adapting to context dynamically.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import random
import re
import base64
import uvicorn
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

load_dotenv()

from openai import AsyncOpenAI

# Load templates
_template_dir = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(_template_dir), trim_blocks=True, lstrip_blocks=True)
_system_template = _env.get_template("system.j2")
_user_template = _env.get_template("user.j2")
_analyzer_system = _env.get_template("analyzer_system.j2")
_analyzer_user = _env.get_template("analyzer_user.j2")
_strategist_system = _env.get_template("strategist_system.j2")
_strategist_user = _env.get_template("strategist_user.j2")
_crafter_system = _env.get_template("crafter_system.j2")
_crafter_user = _env.get_template("crafter_user.j2")
_refiner_system = _env.get_template("refiner_system.j2")
_refiner_user = _env.get_template("refiner_user.j2")
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard, AgentCapabilities, TaskState,
    InvalidParamsError, UnsupportedOperationError, Task
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attacker")


@dataclass
class ScenarioContext:
    """Scenario context from orchestrator for attack generation."""
    attack_type: str                        # INJECTION (embedded) or DIRECT (message)
    domain_description: str                 # Domain/system being attacked
    attack_goal: str                        # What attacker is trying to achieve
    attack_constraints: list[str]           # Rules attacker must follow
    max_number_of_rounds: int               # Maximum battle rounds
    defender_role: str                      # Defender's role description
    defender_task: str                      # What defender does
    output_format: str                      # Expected defender output format
    input_format_template: str              # Template with PLACE_ATTACK_HERE
    current_round: int = 0                  # Current round (0-indexed)
    battle_history: list[dict] = field(default_factory=list)  # Previous rounds

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioContext":
        return cls(
            attack_type=d.get("attack_type", "DIRECT"),
            domain_description=d.get("domain_description", ""),
            attack_goal=d.get("attack_goal", ""),
            attack_constraints=d.get("attack_constraints", []),
            max_number_of_rounds=d.get("max_number_of_rounds", 5),
            defender_role=d.get("defender_role", ""),
            defender_task=d.get("defender_task", ""),
            output_format=d.get("output_format", ""),
            input_format_template=d.get("input_format_template", ""),
            current_round=d.get("current_round", 0),
            battle_history=d.get("battle_history", []),
        )


class Attacker(AgentExecutor):
    """Red team attacker that generates strategic attacks based on scenario context."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model = model

        # Example: Store custom data across rounds. Participants can extend this
        # to track strategy, patterns, or any data useful for future attacks.
        # Keyed by round number (0-indexed). Agent restarts with each battle.
        self.round_data: dict[int, dict] = {}

    def _render(self, template, ctx: ScenarioContext, **extra) -> str:
        """Render a Jinja2 template with the scenario context and extra data."""
        return template.render(**vars(ctx), **extra)
    
    def _apply_cpu_obfuscation(self, payload: str, attack_type: str) -> str:
        """Use local CPU to mutate the payload, bypassing simple keyword filters."""
        # 1. Zero-width space insertion (bypasses exact string matching)
        # We insert a zero-width space (\u200b) randomly inside long words
        words = payload.split()
        obfuscated_words = []
        for word in words:
            if len(word) > 5 and random.random() > 0.5:
                insert_idx = random.randint(1, len(word) - 1)
                word = word[:insert_idx] + '\u200b' + word[insert_idx:]
            obfuscated_words.append(word)
        payload = " ".join(obfuscated_words)

        # 2. Context Smuggling for INJECTIONS
        if attack_type == "INJECTION":
            smuggling_wrappers = [
                f"",
                f"```json\n{{\"data\": \"{payload}\"}}\n```",
                f"| Info | Details |\n|---|---|\n| Note | {payload} |"
            ]
            payload = random.choice(smuggling_wrappers)

        return payload

    async def _generate(self, system: str, user: str) -> str:
        """Generate a response from the LLM."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        content = response.choices[0].message.content
        return content if content is not None else ""

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Generate an attack using a 4-pass reasoning pipeline."""
        if not context.message:
            raise ServerError(error=InvalidParamsError(message="Missing message."))

        task = new_task(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            message_text = context.get_user_input()
            ctx = ScenarioContext.from_dict(json.loads(message_text))
            stored_rounds = dict(self.round_data)

            logger.info("Round %d Pipeline Initiated. History: %d", ctx.current_round + 1, len(ctx.battle_history))

            # ==========================================
            # PASS 1: DEFENSE ANALYZER (API Call 1 of 4)
            # ==========================================
            analyzer_sys = self._render(_analyzer_system, ctx)
            analyzer_usr = self._render(_analyzer_user, ctx, stored_rounds=stored_rounds)
            analysis_result = await self._generate(analyzer_sys, analyzer_usr)
            
            # ==========================================
            # PASS 2: STRATEGIST (API Call 2 of 4)
            # ==========================================
            strategist_sys = self._render(_strategist_system, ctx)
            strategist_usr = self._render(_strategist_user, ctx, analysis=analysis_result)
            strategy_result = await self._generate(strategist_sys, strategist_usr)

            # ==========================================
            # PASS 3: PAYLOAD CRAFTER (API Call 3 of 4)
            # ==========================================
            logger.info("Pass 3: Crafting raw payload...")
            crafter_sys = self._render(_crafter_system, ctx)
            crafter_usr = self._render(_crafter_user, ctx, strategies=strategy_result)
            raw_payload = await self._generate(crafter_sys, crafter_usr)

            # ==========================================
            # PASS 4: REFINER (API Call 4 of 4)
            # ==========================================
            logger.info("Pass 4: Refining and ensuring constraint compliance...")
            refiner_sys = self._render(_refiner_system, ctx)
            refiner_usr = self._render(_refiner_user, ctx, draft_payload=raw_payload)
            final_llm_payload = await self._generate(refiner_sys, refiner_usr)

            # ==========================================
            # CPU OBFUSCATION (Using 10 CPUs limit)
            # ==========================================
            logger.info("Applying Python-based heuristic obfuscation...")
            final_payload = self._apply_cpu_obfuscation(final_llm_payload, ctx.attack_type)

            # Store rich data for the next round
            self.round_data[ctx.current_round] = {
                "analysis": analysis_result,
                "strategy_chosen": strategy_result,
                "final_payload_preview": final_payload[:50] + "..."
            }

            logger.info("Generated attack (%d chars)", len(final_payload))
            await updater.update_status(TaskState.completed, new_agent_text_message(final_payload))
            
        except Exception as e:
            logger.error("Attacker error: %s", e, exc_info=True)
            await updater.update_status(TaskState.failed, new_agent_text_message(f"Error: {e}"))

    async def cancel(self, _request: RequestContext, _event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())


def main():
    parser = argparse.ArgumentParser(description="Security Arena - Attacker Agent")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9021, help="Port to bind to")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()

    agent_card = AgentCard(
        name="attacker",
        description="Red team attacker for security testing",
        url=f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[]
    )

    executor = Attacker(model=args.model)
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(agent_executor=executor, task_store=task_store)
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

    print(f"Starting Attacker on http://{args.host}:{args.port} (model: {args.model})")
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
