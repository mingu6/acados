"""Load a ``ContactImplicitMPC.jl`` flamingo gait (``.jld2``) as numpy arrays.

The file is HDF5 with Julia object references.  ``:split_traj_alt`` keys
(``trajectory.jl:169-179``): ``qm`` (T+2 configurations, absolute angles),
``um`` (T torque impulses, Julia order), ``γm`` (T normal impulses, 4),
``bm`` (T split tangential impulses, 8 as [b+, b-] per contact), ``ψm`` (T),
``ηm`` (T, 8), ``μm``, ``hm``.  Impulses are force times ``hm``.

The default gait is downloaded from ContactImplicitMPC.jl at a pinned
revision, checked against its SHA-256, and cached in ``data/``; set
``CIMPC_DIR`` to a local ContactImplicitMPC.jl clone to read it from there.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from flamingo_model import (
    absolute_to_relative, julia_torque_to_pinocchio,
)

CIMPC_REPOSITORY = "dojo-sim/ContactImplicitMPC.jl"
CIMPC_REVISION = "989c8e6d9675a00c1e342d80707007e82fbcb109"
DEFAULT_GAIT = "src/dynamics/flamingo/gaits/gait_forward_36_4.jld2"
DEFAULT_GAIT_SHA256 = "8e7b01ce970e8f8335a71073e6b5dd88f0629a369a20ddf32efef7fb0193e7bc"
GAIT_CACHE = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class FlamingoGait:
    """Absolute-coordinate gait exactly as stored, plus relative conversions."""

    q_abs: np.ndarray  # (T+2, 9)
    u_julia: np.ndarray  # (T, 6) torque impulses, Julia order
    gamma: np.ndarray  # (T, 4) normal impulses [toe1, heel1, toe2, heel2]
    b: np.ndarray  # (T, 8) [b+, b-] per contact
    psi: np.ndarray  # (T, 4)
    eta: np.ndarray  # (T, 8)
    mu: float
    h: float

    @property
    def horizon(self) -> int:
        return self.u_julia.shape[0]

    @property
    def q(self) -> np.ndarray:
        """(T+2, 9) relative Pinocchio configurations."""
        return np.array([absolute_to_relative(row) for row in self.q_abs])

    @property
    def torque(self) -> np.ndarray:
        """(T, 6) torques in Pinocchio order (impulse divided by ``h``)."""
        return np.array([julia_torque_to_pinocchio(row) for row in self.u_julia]) / self.h

    @property
    def tangential_impulse(self) -> np.ndarray:
        """(T, 4) signed tangential impulse ``b+ - b-`` per contact."""
        return self.b[:, 0::2] - self.b[:, 1::2]

    @property
    def stride(self) -> float:
        return float(self.q_abs[-2, 0] - self.q_abs[0, 0])


def _dereference(file: h5py.File, dataset: str) -> np.ndarray:
    refs = file[dataset][()]
    return np.array([np.asarray(file[ref][()], dtype=float) for ref in refs])


def default_gait_path() -> Path:
    """``$CIMPC_DIR/<gait>`` if set, else the cached download of the pinned gait."""
    if "CIMPC_DIR" in os.environ:
        return Path(os.environ["CIMPC_DIR"]) / DEFAULT_GAIT
    path = GAIT_CACHE / Path(DEFAULT_GAIT).name
    if path.is_file():
        return path
    url = (f"https://raw.githubusercontent.com/{CIMPC_REPOSITORY}/"
           f"{CIMPC_REVISION}/{DEFAULT_GAIT}")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
    except OSError as error:
        raise FileNotFoundError(f"could not download the flamingo gait from {url}: {error}; "
                                "set CIMPC_DIR to a ContactImplicitMPC.jl clone") from error
    digest = hashlib.sha256(content).hexdigest()
    if digest != DEFAULT_GAIT_SHA256:
        raise ValueError(f"flamingo gait from {url} has SHA-256 {digest}, "
                         f"expected {DEFAULT_GAIT_SHA256}")
    GAIT_CACHE.mkdir(exist_ok=True)
    path.write_bytes(content)
    return path


def load_gait(path: str | Path | None = None) -> FlamingoGait:
    path = Path(path) if path is not None else default_gait_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"flamingo gait not found at {path}; set CIMPC_DIR to a "
            "ContactImplicitMPC.jl clone")
    with h5py.File(path, "r") as file:
        gait = FlamingoGait(
            q_abs=_dereference(file, "qm"),
            u_julia=_dereference(file, "um"),
            gamma=_dereference(file, "γm"),
            b=_dereference(file, "bm"),
            psi=_dereference(file, "ψm"),
            eta=_dereference(file, "ηm"),
            mu=float(file["μm"][()]),
            h=float(file["hm"][()]),
        )
    T = gait.horizon
    if gait.q_abs.shape != (T + 2, 9) or gait.gamma.shape != (T, 4) \
            or gait.b.shape != (T, 8) or gait.psi.shape != (T, 4) \
            or gait.eta.shape != (T, 8) or gait.u_julia.shape != (T, 6):
        raise ValueError("unexpected flamingo gait dimensions")
    if not (np.isfinite(gait.h) and gait.h > 0.0 and gait.mu > 0.0):
        raise ValueError("flamingo gait step or friction is invalid")
    return gait
