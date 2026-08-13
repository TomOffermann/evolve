"""Population-parallel execution.

ES is unusually well suited to parallelism, and for a specific structural reason:
**workers exchange (seed, fitness), not gradients.** A few bytes per member per
generation, against gigabytes per step for data-parallel SGD. A heterogeneous pile
of old GPUs on a mediocre interconnect is close to the worst case for SGD and close
to the best case for ES -- see P3.

Three levels of parallelism, all live here:

1. **Within a chunk** -- the population dimension is a batch dimension in every
   einsum. This is EGGROLL's own trick and it is always on.
2. **Across chunks** -- `Executor.map` runs chunks concurrently. Chunking also
   bounds peak memory, which is what allows N far larger than fits at once.
3. **Across processes/nodes** -- not implemented here, but the contract is fixed by
   `ChunkResult`: a worker needs only `(seed, chunk_id, sigma, rank, generation)` to
   reconstruct its slice of the population, and returns only fitness tensors. Any
   transport that can move those can run distributed ES.

`ThreadExecutor` is genuinely useful despite the GIL because torch releases it
during matmuls; for small models the launch overhead dominates and `SerialExecutor`
wins, so the default is chosen by size, not by faith.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ChunkSpec:
    """Everything a worker needs to reconstruct and evaluate its slice.

    Deliberately picklable and tiny -- this is the distributed wire format."""

    chunk_id: int
    n_chunks: int
    lo: int
    hi: int
    seed: int
    crn_seed: int
    sigma: float
    rank: int
    generation: int

    @property
    def size(self) -> int:
        return self.hi - self.lo


def plan_chunks(n_pop: int, chunk_size: int | None, seed: int, crn_seed: int,
                sigma: float, rank: int, generation: int) -> list[ChunkSpec]:
    cs = n_pop if not chunk_size else min(chunk_size, n_pop)
    bounds = [(lo, min(lo + cs, n_pop)) for lo in range(0, n_pop, cs)]
    return [
        ChunkSpec(i, len(bounds), lo, hi, seed, crn_seed, sigma, rank, generation)
        for i, (lo, hi) in enumerate(bounds)
    ]


class Executor(Protocol):
    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]: ...
    def shutdown(self) -> None: ...


class SerialExecutor:
    """One chunk at a time. Lowest overhead; correct by construction."""

    def map(self, fn, items):
        return [fn(x) for x in items]

    def shutdown(self):
        pass


class ThreadExecutor:
    """Concurrent chunks. Useful because torch releases the GIL inside ops.

    Objectives must be thread-safe for *reads* of their parameters. The trainer
    guarantees no writes happen during evaluation."""

    def __init__(self, workers: int = 4):
        self.workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def map(self, fn, items):
        return list(self._pool.map(fn, items))

    def shutdown(self):
        self._pool.shutdown(wait=False)


def auto_executor(n_pop: int, chunk_size: int | None, workers: int = 4) -> Executor:
    """Threads only when there is enough work per chunk to amortise the launch.

    Below roughly 4 chunks the pool costs more than it saves, which is why this
    picks by size rather than defaulting to parallel everywhere."""
    if not chunk_size or n_pop <= chunk_size * 3:
        return SerialExecutor()
    return ThreadExecutor(workers)
