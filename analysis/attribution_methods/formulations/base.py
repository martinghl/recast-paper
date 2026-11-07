"""Core abstractions: a Formulation = (Target x Baseline x SignPolicy)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np
import torch


@runtime_checkable
class Target(Protocol):
    name: str
    needs_references: bool
    def reference(self, ctx: "EngineContext", inputs: torch.Tensor,
                  labels_batch: np.ndarray) -> torch.Tensor: ...
    def scalar(self, e_x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class Baseline(Protocol):
    name: str
    needs_references: bool
    def vector(self, ctx: "EngineContext", inputs: torch.Tensor,
               labels_batch: np.ndarray) -> torch.Tensor: ...


@runtime_checkable
class SignPolicy(Protocol):
    name: str
    def apply(self, signed_attr: np.ndarray) -> dict: ...   # {suffix: non-negative ndarray}


@dataclass
class ClusterReferences:
    clusters: list
    prototypes: dict          # cluster -> unit np.ndarray [latent]
    competitor: dict          # cluster -> nearest-other cluster
    centroid_lognorm: dict    # cluster -> np.ndarray [n_genes] (mean lognorm)


@dataclass
class EngineContext:
    encoder: object
    device: str
    references: ClusterReferences | None = None


@dataclass
class Formulation:
    target: Target
    baseline: Baseline
    sign: SignPolicy

    @property
    def name(self) -> str:
        return f"{self.target.name}__{self.baseline.name}__{self.sign.name}"

    @property
    def needs_references(self) -> bool:
        return bool(self.target.needs_references or self.baseline.needs_references)
