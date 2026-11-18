import json
import os

import numpy as np
import pytest

from ase.atoms import Atoms
from ase.build import add_adsorbate, bulk, fcc111

from symmetrix import _mace_foundation

try:
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    from symmetrix.extract_mace_data import extract_mace_data
except ImportError:
    download_mace_mp_checkpoint = None


def structures():
    """a mixed batch: non-orthogonal bulk water, a water slab, a water cluster, bulk Cu and a Cu(111)/water cell"""
    water = Atoms(
        "OH2",
        positions=[[0.7, 0.9, 1.1], [1.66, 0.9, 1.1], [0.46, 1.83, 1.1]],
        cell=[[3.1, 0, 0], [0.4, 3.0, 0], [0.3, -0.5, 3.2]],
        pbc=True,
    ).repeat((2, 2, 2))
    water.rattle(0.05, seed=1)
    slab = water.copy()
    slab.pbc = [True, True, False]
    cluster = Atoms(water.numbers, positions=water.positions)
    cu = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    cu.rattle(0.05, seed=2)
    interface = fcc111("Cu", (2, 2, 3), vacuum=5.0, periodic=True)
    h2o = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    add_adsorbate(interface, h2o, 2.3, "ontop")
    add_adsorbate(interface, h2o, 5.2, "fcc")
    interface.rattle(0.03, seed=3)
    return [water, slab, cluster, cu, interface]


@pytest.fixture(scope="module")
def omat_small_cu(tmp_path_factory):
    if download_mace_mp_checkpoint is None:
        pytest.skip("mace-torch is not available")
    cache_dir = tmp_path_factory.mktemp("mace_cache")
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    model = str(download_mace_mp_checkpoint("small-omat-0"))
    if xdg_cache_home is None:
        del os.environ["XDG_CACHE_HOME"]
    else:
        os.environ["XDG_CACHE_HOME"] = xdg_cache_home
    model_json = cache_dir / "small-omat-0-1-8-29.json"
    model_json.write_text(json.dumps(extract_mace_data(model, [1, 8, 29])))
    return str(model_json)


def evaluator(model_json, device, dtype):
    try:
        return _mace_foundation.MACEFoundation(model_json, device, dtype, 1.0)
    except RuntimeError as e:
        no_gpu = ["built without CUDA", "no CUDA-capable device", "driver version is"]
        if device == "cuda" and any(m in str(e) for m in no_gpu):
            pytest.skip(str(e))
        raise


def batch_edges(atoms, cutoff):
    """concatenated edges of the structures, from neighbor lists built one structure at a time"""
    i, j, shift, system = [], [], [], []
    o = 0
    for s, a in enumerate(atoms):
        p, q, S = _mace_foundation.neighbor_list(
            a.positions, a.cell.array, a.pbc, cutoff
        )
        i.append(p + o)
        j.append(q + o)
        shift.append(S @ a.cell.array)
        system.append(np.full(len(a), s))
        o += len(a)
    return [np.concatenate(v) for v in (i, j, shift, system)]


TOL = {  # energy per atom, forces, virial per atom
    "float64": (1e-10, 1e-10, 1e-10),
    "float32": (1e-5, 1e-4, 1e-4),
    "float32-mixed": (1e-5, 1e-4, 1e-4),
}


def check(atoms, batch, reference, dtype):
    tol_e, tol_f, tol_v = TOL[dtype]
    energy, forces, virial, energies = batch
    first = np.cumsum([0] + [len(a) for a in atoms])
    for s, (a, (e, f, v, ea)) in enumerate(zip(atoms, reference)):
        n = len(a)
        assert abs(energy[s] - e) / n < tol_e
        assert np.abs(forces[first[s] : first[s + 1]] - f).max() < tol_f
        assert np.abs(virial[s] - v).max() / n < tol_v
        assert np.abs(energies[first[s] : first[s + 1]] - ea).max() < tol_e * 10


@pytest.mark.parametrize(
    "device, dtype",
    [
        ("cpu", "float64"),
        ("cpu", "float32"),
        ("cuda", "float64"),
        ("cuda", "float32"),
        ("cuda", "float32-mixed"),
    ],
)
@pytest.mark.parametrize("skin", [0.0, 1.0])
def test_batch_vs_compute(omat_small_cu, device, dtype, skin):
    atoms = structures()
    ev = evaluator(omat_small_cu, device, dtype)
    reference = [
        evaluator(omat_small_cu, device, dtype).compute(
            a.positions, a.numbers, a.cell.array, a.pbc, True
        )
        for a in atoms
    ]
    i, j, shift, system = batch_edges(atoms, ev.r_cut + skin)
    x = np.concatenate([a.positions for a in atoms])
    numbers = np.concatenate([a.numbers for a in atoms])
    batch = ev.compute_batch(x, numbers, i, j, shift, system, len(atoms), True)
    assert batch[0].shape == (len(atoms),) and batch[2].shape == (len(atoms), 6)
    check(atoms, batch, reference, dtype)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batch_list_reuse(omat_small_cu, device):
    """edges built with a skin stay valid while atoms move less than skin/2; compute and compute_batch alternate"""
    dtype = "float64"
    atoms = structures()
    ev = evaluator(omat_small_cu, device, dtype)
    i, j, shift, system = batch_edges(atoms, ev.r_cut + 1.0)
    numbers = np.concatenate([a.numbers for a in atoms])
    rng = np.random.default_rng(0)
    for step in range(3):
        for a in atoms:
            d = rng.normal(0, 0.05, a.positions.shape)
            a.positions += d * min(
                1, 0.15 / np.linalg.norm(d, axis=1).max()
            )  # < skin/2 in total
        reference = [
            evaluator(omat_small_cu, device, dtype).compute(
                a.positions, a.numbers, a.cell.array, a.pbc, True
            )
            for a in atoms
        ]
        assert ev.compute(
            atoms[step].positions,
            atoms[step].numbers,
            atoms[step].cell.array,
            atoms[step].pbc,
        )[0] == pytest.approx(reference[step][0], abs=1e-9)
        x = np.concatenate([a.positions for a in atoms])
        batch = ev.compute_batch(x, numbers, i, j, shift, system, len(atoms), True)
        check(atoms, batch, reference, dtype)
    # a smaller batch on the same evaluator
    sub = atoms[3:]
    i, j, shift, system = batch_edges(sub, ev.r_cut)
    x = np.concatenate([a.positions for a in sub])
    numbers = np.concatenate([a.numbers for a in sub])
    batch = ev.compute_batch(x, numbers, i, j, shift, system, len(sub), True)
    check(sub, batch, reference[3:], dtype)


def test_batch_errors(omat_small_cu):
    atoms = structures()[:2]
    ev = evaluator(omat_small_cu, "cpu", "float64")
    i, j, shift, system = batch_edges(atoms, ev.r_cut)
    x = np.concatenate([a.positions for a in atoms])
    numbers = np.concatenate([a.numbers for a in atoms])
    with pytest.raises(ValueError, match="joins two systems"):
        ev.compute_batch(x, numbers, i, j, shift, np.arange(len(x)) % 2)
    with pytest.raises(ValueError, match="ascending"):
        ev.compute_batch(x, numbers, i[::-1], j[::-1], shift[::-1], system)
    with pytest.raises(ValueError, match="num_systems"):
        ev.compute_batch(x, numbers, i, j, shift, system, 1)
    # one system by default
    n = len(atoms[0])
    keep = i < n
    e = ev.compute_batch(x[:n], numbers[:n], i[keep], j[keep], shift[keep])[0]
    ref = ev.compute(
        atoms[0].positions, atoms[0].numbers, atoms[0].cell.array, atoms[0].pbc
    )
    assert e.shape == (1,) and e[0] == pytest.approx(ref[0], abs=1e-9)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_sim(omat_small_cu, device):
    torch = pytest.importorskip("torch")
    ts = pytest.importorskip("torch_sim")
    from symmetrix.experimental import MACEFoundation
    from symmetrix.torch_sim import SymmetrixMACEFoundationModel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    evaluator(omat_small_cu, device, "float64")  # skips without a CUDA build
    with pytest.warns(UserWarning, match="experimental"):
        model = SymmetrixMACEFoundationModel(omat_small_cu, device=device)
    atoms = [a for a in structures() if a.pbc.all()]
    state = ts.io.atoms_to_state(atoms, model.device, torch.float64)
    out = model(state)
    assert out["energy"].shape == (len(atoms),)
    assert out["forces"].shape == (state.n_atoms, 3)
    assert out["stress"].shape == (len(atoms), 3, 3)
    first = np.cumsum([0] + [len(a) for a in atoms])
    for s, a in enumerate(atoms):
        with pytest.warns(UserWarning, match="experimental"):
            a.calc = MACEFoundation(omat_small_cu, device=device, dtype="float64")
        assert out["energy"][s].item() == pytest.approx(
            a.get_potential_energy(), abs=1e-8
        )
        f = out["forces"][first[s] : first[s + 1]].cpu().numpy()
        assert np.abs(f - a.get_forces()).max() < 1e-9
        stress = out["stress"][s].cpu().numpy()
        assert np.abs(stress - stress.T).max() < 1e-12
        assert (
            np.abs(
                stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] - a.get_stress()
            ).max()
            < 1e-9
        )

    final = ts.integrate(
        system=state,
        model=model,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=10,
        temperature=300.0,
        timestep=0.001,
    )
    assert final.n_systems == len(atoms)
    assert torch.isfinite(final.positions).all()
    out = model(final)
    assert torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all()
    # the batch matches the structures evaluated alone after the dynamics too
    for s in range(len(atoms)):
        alone = model(final[s])
        assert alone["energy"].item() == pytest.approx(
            out["energy"][s].item(), abs=1e-8
        )
