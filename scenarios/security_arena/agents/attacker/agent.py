#!/usr/bin/env python3
"""
Attacker Agent v9 — Generic Adaptive Red Team Agent.

Architecture:
  Routes each battle to the appropriate specialist agent based on attack type:
  - INJECTION: InjectionAttackAgent (payload embedded in document data)
  - DIRECT:    DirectAttackAgent    (message sent directly to stateless defender)

  Each specialist uses a 4-LLM-call pipeline per round:
    Round 0:  3 generators + 1 judge = 4 calls
    Rounds 1+: 1 summarizer + 2 generators + 1 judge = 4 calls
"""

import argparse
import json
import logging

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    TaskState,
    InvalidParamsError,
    UnsupportedOperationError,
    Task,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

try:
    from .shared import ScenarioContext
    from .injection_attack_agent import InjectionAttackAgent
    from .direct_attack_agent import DirectAttackAgent
except ImportError:
    from shared import ScenarioContext
    from injection_attack_agent import InjectionAttackAgent
    from direct_attack_agent import DirectAttackAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attacker")


# ---------------------------------------------------------------------------
# Attacker — routes to specialist agent based on attack type
# ---------------------------------------------------------------------------
class Attacker(AgentExecutor):

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self._injection_agent = InjectionAttackAgent(model)
        self._direct_agent = DirectAttackAgent(model)

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        if not context.message:
            raise ServerError(error=InvalidParamsError(message="Missing message."))

        task = new_task(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            message_text = context.get_user_input()
            ctx = ScenarioContext.from_dict(json.loads(message_text))
            logger.info(
                "Round %d/%d: Type=%s Goal=%s",
                ctx.current_round + 1,
                ctx.max_number_of_rounds,
                ctx.attack_type,
                ctx.attack_goal[:80],
            )

            if ctx.attack_type.upper() == "INJECTION":
                output = await self._injection_agent.execute(ctx)
            else:
                output = await self._direct_agent.execute(ctx)

            logger.info("Output: %d chars, first 100: %s", len(output), output[:100])

            await updater.update_status(
                TaskState.completed, new_agent_text_message(output)
            )
        except Exception as e:
            logger.error("Attacker error: %s", e, exc_info=True)
            await updater.update_status(
                TaskState.failed, new_agent_text_message(f"Error: {e}")
            )

    async def cancel(self, _req: RequestContext, _eq: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9021)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    args = parser.parse_args()

    agent_card = AgentCard(
        name="attacker",
        description="Generic adaptive red team attacker v9 (injection/direct specialist routing)",
        url=f"http://{args.host}:{args.port}/",
        version="9.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )

    executor = Attacker(model=args.model)
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor, task_store=task_store
    )
    app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    print(f"Attacker v9 on http://{args.host}:{args.port} (model: {args.model})")
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
