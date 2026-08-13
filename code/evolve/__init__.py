"""evolve -- low-rank Evolution Strategies with pluggable EA operators.

    from evolve import Trainer, TrainConfig, registry
    from evolve.objectives.countdown_obj import CountdownObjective

    obj = CountdownObjective(seed=0)
    tr = Trainer(
        obj,
        sampler   = registry.build("sampling",  "iid"),
        weighting = registry.build("weighting", "novelty", lam=0.2),
        config    = TrainConfig(n_pop=128, sigma=0.005, alpha=0.001, generations=250),
        sigma_rule= registry.build("sigma", "resolution", sigma=0.005),
        diagnostics = default_diagnostics(128),
    )
    tr.run()
    print(obj.report())

`registry.describe()` prints every operator with its measured verdict. Read it
before picking one: E0-E4 established that whether an operator helps depends on the
regime, not on the operator.
"""

from .core.trainer import GenerationRecord, TrainConfig, Trainer
from .core.types import EvalRequest, Evaluation
from .diagnostics.adapters import default_diagnostics

__all__ = [
    "Trainer", "TrainConfig", "GenerationRecord",
    "EvalRequest", "Evaluation", "default_diagnostics",
]
