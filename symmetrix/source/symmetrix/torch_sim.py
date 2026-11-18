"""TorchSim model for the models of pair_style symmetrix/mace/foundation (experimental)

Each structure of a SimState batch gets a neighbor list of its own, from symmetrix's neighbor_list, and the batch is
evaluated as one disconnected graph by a single MACEFoundation.compute_batch call.
"""

import warnings

import numpy as np
import torch

try:
    from torch_sim.models.interface import ModelInterface
except ImportError as e:
    raise ImportError(
        "symmetrix.torch_sim needs TorchSim: pip install torch-sim-atomistic"
    ) from e


class SymmetrixMACEFoundationModel(ModelInterface):
    """The models of pair_style symmetrix/mace/foundation as a TorchSim ModelInterface.

    Parameters
    ----------
    model_file: str
        Symmetrix JSON of the model, as for symmetrix.experimental.MACEFoundation.
    device: torch.device or str
        Device of the returned tensors. The evaluator runs on the GPU when this is a CUDA device (positions and
        results pass through host memory) and on the CPU otherwise. Defaults to CUDA when available.
    dtype: torch.dtype
        dtype of the returned tensors, normally that of the states. Defaults to torch.float64.
    evaluator_dtype: "float64", "float32" or "float32-mixed"
        Arithmetic of the evaluator, as for MACEFoundation.
    skin: float
        The neighbor list of a structure (cutoff r_cut + skin) is built again when its cell changes or one of its
        atoms has moved by more than skin/2, and all of them when the atoms or systems of the batch change.
    compute_forces, compute_stress: bool
        Whether forward returns "forces" and "stress". The stress is -virial/volume (eV/A^3, the sign of ASE and
        TorchSim), and zero for a structure whose cell has no volume.
    """

    def __init__(
        self,
        model_file,
        *,
        device=None,
        dtype=torch.float64,
        evaluator_dtype="float64",
        skin=1.0,
        compute_forces=True,
        compute_stress=True,
    ):
        from ._mace_foundation import MACEFoundation, neighbor_list

        warnings.warn(
            "SymmetrixMACEFoundationModel is experimental: its interface, model format and results may "
            "change between releases",
            stacklevel=2,
        )
        super().__init__()
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self._compute_forces = bool(compute_forces)
        self._compute_stress = bool(compute_stress)
        self._memory_scales_with = "n_atoms_x_density"
        self.evaluator = MACEFoundation(
            str(model_file),
            "cuda" if self._device.type == "cuda" else "cpu",
            evaluator_dtype,
            skin,
        )
        self._neighbor_list = neighbor_list
        self.skin = float(skin)
        self.cutoff = float(self.evaluator.r_cut) + self.skin
        self._lists = None

    @ModelInterface.compute_stress.setter
    def compute_stress(self, value):
        self._compute_stress = bool(value)

    @ModelInterface.compute_forces.setter
    def compute_forces(self, value):
        self._compute_forces = bool(value)

    def _edges(self, x, numbers, cells, pbcs, system, first):
        """i, j and shifts (r_ij = x[j] - x[i] + shift) of the batch, from the lists that are still valid"""
        layout = (numbers.tobytes(), first.tobytes(), pbcs.tobytes())
        c = self._lists
        if c is None or c["layout"] != layout:
            c = self._lists = {
                "layout": layout,
                "parts": [None] * len(cells),
                "x": x.copy(),
                "cells": cells.copy(),
            }
            stale = range(len(cells))
        else:
            d = x - c["x"]
            moved = 4 * np.einsum("ij,ij->i", d, d) > self.skin**2
            changed = (cells != c["cells"]).any(axis=(1, 2))
            stale = np.union1d(system[moved], np.flatnonzero(changed))
        for s in stale:
            a, b = first[s], first[s + 1]
            i, j, S = self._neighbor_list(x[a:b], cells[s], pbcs[s], self.cutoff)
            c["parts"][s] = (i + a, j + a, np.asarray(S, dtype=np.float64) @ cells[s])
            c["x"][a:b] = x[a:b]
            c["cells"][s] = cells[s]
        if len(stale):
            c["i"], c["j"], c["shift"] = (
                np.ascontiguousarray(np.concatenate(p)) for p in zip(*c["parts"])
            )
        return c["i"], c["j"], c["shift"]

    def forward(self, state, **kwargs):
        """energy [n_systems], forces [n_atoms, 3] and stress [n_systems, 3, 3] of the structures of state"""
        x = np.ascontiguousarray(
            state.positions.detach().to("cpu", torch.float64).numpy()
        )
        numbers = state.atomic_numbers.detach().to("cpu", torch.int32).numpy()
        system = state.system_idx.detach().to("cpu", torch.int32).numpy()
        # TorchSim stores the cell vectors as columns, symmetrix (like ASE) as rows
        cells = np.ascontiguousarray(
            state.cell.detach().to("cpu", torch.float64).mT.numpy()
        )
        n_systems = len(cells)
        pbcs = np.ascontiguousarray(
            np.broadcast_to(
                state.pbc.detach().cpu().numpy().astype(bool), (n_systems, 3)
            )
        )
        counts = np.bincount(system, minlength=n_systems)
        if len(counts) != n_systems or np.any(np.diff(system) < 0):
            raise ValueError("system_idx must be sorted, with values in [0, n_systems)")
        first = np.concatenate([[0], np.cumsum(counts)])
        i, j, shift = self._edges(x, numbers, cells, pbcs, system, first)
        energy, forces, virial, _ = self.evaluator.compute_batch(
            x, numbers, i, j, shift, system, n_systems
        )
        out = {
            "energy": torch.as_tensor(energy, dtype=self._dtype, device=self._device)
        }
        if self._compute_forces:
            out["forces"] = torch.as_tensor(
                forces, dtype=self._dtype, device=self._device
            )
        if self._compute_stress:
            # virial in LAMMPS order xx yy zz xy xz yz
            full = virial[:, [0, 3, 4, 3, 1, 5, 4, 5, 2]].reshape(-1, 3, 3)
            volume = np.abs(np.linalg.det(cells))
            stress = np.zeros_like(full)
            ok = volume > 0
            stress[ok] = -full[ok] / volume[ok, None, None]
            out["stress"] = torch.as_tensor(
                stress, dtype=self._dtype, device=self._device
            )
        return out
