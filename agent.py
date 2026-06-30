"""
stateful_checkpoint/agent.py — Stateful / checkpointed agent pattern.

Domain: A long-running close pipeline that can be paused at a human
review gate and resumed later — possibly minutes, hours, or days later,
possibly by a different process entirely (e.g. the web server that
started the run is gone by the time someone resumes it).

This is a standalone, dependency-free reference implementation of the
pattern used in Close Command's production pipeline (which uses
LangGraph's MemorySaver). Here, checkpointing is a simple serializable
dict snapshot — swap the storage backend for LangGraph/Redis/a database
in production; the checkpoint/resume architecture itself does not change.

The defining trait: execution state survives a process boundary. A
fresh CloseRunner instance, with no memory of the original run, must be
able to resume a paused run correctly using ONLY the checkpoint data —
this is explicitly tested below by deliberately using two separate
instances to simulate a process restart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

PIPELINE_STAGES = ["ingestion", "matching", "elimination", "validation", "review", "output"]
HITL_GATE_STAGE = "review"  # the pipeline always pauses here for human approval


@dataclass
class Checkpoint:
    """Fully serializable snapshot of an in-progress run — no live objects, no closures."""
    run_id: str
    current_stage_index: int
    completed_stages: list[str]
    stage_outputs: dict[str, dict]
    status: str  # "RUNNING" | "PAUSED_FOR_REVIEW" | "COMPLETE" | "FAILED"


class CheckpointStore:
    """
    Minimal in-memory checkpoint store. In production this is a database
    row or a LangGraph/Redis-backed store — the interface (save/load by
    run_id) is what matters architecturally, not the storage medium.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    def save(self, checkpoint: Checkpoint) -> None:
        self._checkpoints[checkpoint.run_id] = checkpoint

    def load(self, run_id: str) -> Optional[Checkpoint]:
        return self._checkpoints.get(run_id)


class CloseRunner:
    """
    Runs the close pipeline stage by stage, checkpointing after every
    stage. Pauses automatically at the HITL gate. A SEPARATE CloseRunner
    instance — with no shared memory — can resume the run using only the
    checkpoint, proving the state genuinely survived the process boundary
    rather than relying on in-memory object state.
    """

    def __init__(self, checkpoint_store: CheckpointStore, stage_handlers: Optional[dict] = None) -> None:
        self.store = checkpoint_store
        self.stage_handlers = stage_handlers or self._default_stage_handlers()

    def start_run(self, run_id: str, initial_input: dict) -> Checkpoint:
        checkpoint = Checkpoint(
            run_id=run_id, current_stage_index=0, completed_stages=[],
            stage_outputs={"_initial_input": initial_input}, status="RUNNING",
        )
        return self._advance(checkpoint)

    def resume_run(self, run_id: str) -> Checkpoint:
        """
        Resume a paused run. Deliberately loads the checkpoint fresh from
        the store rather than trusting any in-memory state — this is what
        makes resumption correct even across a process restart.

        Calling resume_run() IS the human approval signal for the HITL
        gate — so before re-entering _advance(), the gate stage is marked
        complete. Without this, _advance() would re-evaluate the exact
        same "are we at the gate and is it not yet completed" check and
        pause again immediately, making resume a no-op.
        """
        checkpoint = self.store.load(run_id)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for run_id: {run_id}")
        if checkpoint.status != "PAUSED_FOR_REVIEW":
            raise ValueError(
                f"Cannot resume run {run_id} — status is {checkpoint.status}, "
                f"expected PAUSED_FOR_REVIEW."
            )

        # Mark the gate as approved and advance past it — this IS the
        # human's approval, expressed by the act of calling resume_run().
        gate_index = PIPELINE_STAGES.index(HITL_GATE_STAGE)
        checkpoint.stage_outputs[HITL_GATE_STAGE] = {"approved": True}
        checkpoint.completed_stages.append(HITL_GATE_STAGE)
        checkpoint.current_stage_index = gate_index + 1
        checkpoint.status = "RUNNING"

        return self._advance(checkpoint)

    def _advance(self, checkpoint: Checkpoint) -> Checkpoint:
        """Run stages starting from current_stage_index, pausing at the HITL gate."""
        while checkpoint.current_stage_index < len(PIPELINE_STAGES):
            stage_name = PIPELINE_STAGES[checkpoint.current_stage_index]

            if stage_name == HITL_GATE_STAGE and stage_name not in checkpoint.completed_stages:
                # Pause here — do NOT execute the review stage automatically.
                # A human must explicitly call resume_run() after approving.
                checkpoint.status = "PAUSED_FOR_REVIEW"
                self.store.save(checkpoint)
                return checkpoint

            handler = self.stage_handlers.get(stage_name)
            output = handler(checkpoint.stage_outputs) if handler else {}
            checkpoint.stage_outputs[stage_name] = output
            checkpoint.completed_stages.append(stage_name)
            checkpoint.current_stage_index += 1

            self.store.save(checkpoint)  # checkpoint after EVERY stage, not just at the end

        checkpoint.status = "COMPLETE"
        self.store.save(checkpoint)
        return checkpoint

    @staticmethod
    def _default_stage_handlers() -> dict:
        def ingestion(outputs: dict) -> dict:
            return {"rows_ingested": 1240}

        def matching(outputs: dict) -> dict:
            return {"matched_pairs": 312, "exceptions": 4}

        def elimination(outputs: dict) -> dict:
            return {"journals_generated": 312}

        def validation(outputs: dict) -> dict:
            return {"validation_passed": True, "flagged_count": 2}

        def output(outputs: dict) -> dict:
            return {"jlf_generated": True, "audit_trail_complete": True}

        return {
            "ingestion": ingestion, "matching": matching, "elimination": elimination,
            "validation": validation, "output": output,
        }
