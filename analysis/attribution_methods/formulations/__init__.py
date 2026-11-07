"""Registry: configuration name -> Formulation."""
from .base import Formulation
from .targets import SelfRepr, PrototypeSim, Contrastive, DecisionFn
from .baselines import Zero, Counterfactual
from .sign import AbsSign, SignedSign

_TARGETS = {"self": SelfRepr, "proto": PrototypeSim, "contrast": Contrastive,
            "decision": DecisionFn}
_BASELINES = {"zero": Zero, "cf": Counterfactual}
_SIGNS = {"abs": AbsSign, "signed": SignedSign}


def build_formulation(name: str) -> Formulation:
    t, b, s = name.split("__")
    return Formulation(_TARGETS[t](), _BASELINES[b](), _SIGNS[s]())


# 3 targets x 2 baselines, signed, plus self-embedding / zero baseline / |.| (the reference-free attribution).
V1_GRID = ["self__zero__abs"] + [
    f"{t}__{b}__signed" for t in ("self", "proto", "contrast") for b in ("zero", "cf")
]
