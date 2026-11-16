# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for the "Yanshu debug here" request-lifecycle log lines.

A learning aid, not for merging. Every log call whose message starts with
"Yanshu debug here, [Snn <stage>]" marks one stage on the path of a request:

    S01-S08  API server:  HTTP route -> tokenize -> AsyncLLM -> ZMQ send
    S09-S14  EngineCore:  ZMQ receive -> scheduler -> RPC to the workers
    S15-S18  workers:     execute_model / sample_tokens
    S19-S22  EngineCore:  outputs -> ZMQ send
    S23-S30  API server:  detokenize -> RequestOutput -> HTTP response
    S31-S32  abort

`grep -rn "Yanshu debug here"` lists every site. S16 and S17 are in the
vllm-torchtpu worker.
"""

import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

_seen: set[str] = set()


def stamp() -> str:
    """Return the time with milliseconds, and the thread name.

    vLLM's own log timestamp has whole seconds, which is too coarse to put
    lines from the API server, EngineCore and worker processes in one order.
    """
    now = time.time()
    clock = time.strftime("%H:%M:%S", time.localtime(now))
    millis = int(now * 1000) % 1000
    return f"t={clock}.{millis:03d} thr={threading.current_thread().name}"


def first_time(key: str) -> bool:
    """Return True the first time this process asks about `key`."""
    if key in _seen:
        return False
    _seen.add(key)
    return True


def short_ids(token_ids: Sequence[int] | None, keep: int = 8) -> str:
    """Format a token id list, cut to its first `keep` ids."""
    if token_ids is None:
        return "None"
    ids = list(token_ids)
    if len(ids) <= keep:
        return str(ids)
    return f"{ids[:keep]}+{len(ids) - keep} more"


def is_idle_step(scheduler_output: "SchedulerOutput") -> bool:
    """Return True for a step that schedules nothing and reports nothing finished.

    EngineCore repeats such steps about every millisecond while the scheduler
    holds a request it cannot advance, for example one waiting on a KV
    transfer, so they are not logged.
    """
    return not (
        scheduler_output.total_num_scheduled_tokens or scheduler_output.finished_req_ids
    )


def describe_step(scheduler_output: "SchedulerOutput") -> str:
    """Describe what one SchedulerOutput asks the workers to do."""
    first_time_reqs = ", ".join(
        f"{new_req.req_id}(prompt_tokens={len(new_req.prompt_token_ids or ())} "
        f"already_computed={new_req.num_computed_tokens} "
        f"kv_blocks={sum(len(group) for group in new_req.block_ids)})"
        for new_req in scheduler_output.scheduled_new_reqs
    )
    cached = scheduler_output.scheduled_cached_reqs
    continuing_reqs = ", ".join(
        f"{req_id}(computed={computed})"
        for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens)
    )
    return (
        f"tokens_this_step={scheduler_output.total_num_scheduled_tokens} "
        f"per_request={scheduler_output.num_scheduled_tokens} "
        f"first_time=[{first_time_reqs}] continuing=[{continuing_reqs}] "
        f"finished_since_last_step={sorted(scheduler_output.finished_req_ids)}"
    )
