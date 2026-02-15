"""Tests for Metatomic force provider correctness across DD configurations.

Run with:
    pytest test_metatomic.py -v

Environment variables:
    GMX_BIN:  path to gmx_mpi binary (default: gmx_mpi)
    MPIRUN:   path to mpirun (default: mpirun)

These tests verify that the metatomic force provider gives identical
results regardless of domain decomposition partitioning, and that
energies/forces are physically reasonable.
"""

import shutil

import pytest

from conftest import (
    MDRunResult,
    parse_debug_log,
    run_mdrun,
)

# Reference step-0 energy from verified serial + DD runs
REFERENCE_STEP0_ENERGY = -246.379  # kJ/mol, from previous validation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def serial_run():
    """Run simulation with 1 rank (serial)."""
    result = run_mdrun(nranks=1, nsteps=100, timer=True)
    yield result
    shutil.rmtree(result.workdir, ignore_errors=True)


@pytest.fixture(scope="session")
def dd4_run():
    """Run simulation with 4 DD ranks."""
    result = run_mdrun(nranks=4, nsteps=100, timer=True)
    yield result
    shutil.rmtree(result.workdir, ignore_errors=True)


@pytest.fixture(scope="session")
def dd8_run():
    """Run simulation with 8 DD ranks."""
    result = run_mdrun(nranks=8, nsteps=100, timer=True)
    yield result
    shutil.rmtree(result.workdir, ignore_errors=True)


@pytest.fixture(scope="session")
def dd12_run():
    """Run simulation with 12 DD ranks."""
    result = run_mdrun(nranks=12, nsteps=100, timer=True)
    yield result
    shutil.rmtree(result.workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------


class TestMDRunSuccess:
    """Verify that mdrun completes successfully."""

    def test_serial_exits_zero(self, serial_run: MDRunResult):
        assert serial_run.returncode == 0, f"stderr:\n{serial_run.stderr}"

    def test_dd4_exits_zero(self, dd4_run: MDRunResult):
        assert dd4_run.returncode == 0, f"stderr:\n{dd4_run.stderr}"

    @pytest.mark.dd8
    def test_dd8_exits_zero(self, dd8_run: MDRunResult):
        assert dd8_run.returncode == 0, f"stderr:\n{dd8_run.stderr}"

    @pytest.mark.dd12
    def test_dd12_exits_zero(self, dd12_run: MDRunResult):
        assert dd12_run.returncode == 0, f"stderr:\n{dd12_run.stderr}"


# ---------------------------------------------------------------------------
# Step-0 energy consistency across DD configurations
# ---------------------------------------------------------------------------


class TestStep0Energy:
    """Step-0 energies must be identical across all DD configurations.

    Since step-0 coordinates are identical (from the same .gro file),
    the model evaluation must produce the same total energy regardless
    of how many domains the system is split into.
    """

    def _step0_metatomic(self, result: MDRunResult) -> float:
        assert len(result.mdlog.frames) >= 1, "No energy frames found"
        f = result.mdlog.frames[0]
        assert f.step == 0
        return f.metatomic

    def test_serial_reference(self, serial_run):
        e = self._step0_metatomic(serial_run)
        assert e == pytest.approx(REFERENCE_STEP0_ENERGY, abs=0.1), (
            f"Serial step-0 energy {e:.3f} != reference {REFERENCE_STEP0_ENERGY:.3f}"
        )

    def test_dd4_matches_serial(self, serial_run, dd4_run):
        e_serial = self._step0_metatomic(serial_run)
        e_dd4 = self._step0_metatomic(dd4_run)
        assert e_dd4 == pytest.approx(e_serial, abs=0.01), (
            f"DD4 step-0 energy {e_dd4:.6f} != serial {e_serial:.6f}"
        )

    @pytest.mark.dd8
    def test_dd8_matches_serial(self, serial_run, dd8_run):
        e_serial = self._step0_metatomic(serial_run)
        e_dd8 = self._step0_metatomic(dd8_run)
        assert e_dd8 == pytest.approx(e_serial, abs=0.01), (
            f"DD8 step-0 energy {e_dd8:.6f} != serial {e_serial:.6f}"
        )

    @pytest.mark.dd12
    def test_dd12_matches_serial(self, serial_run, dd12_run):
        e_serial = self._step0_metatomic(serial_run)
        e_dd12 = self._step0_metatomic(dd12_run)
        assert e_dd12 == pytest.approx(e_serial, abs=0.01), (
            f"DD12 step-0 energy {e_dd12:.6f} != serial {e_serial:.6f}"
        )


# ---------------------------------------------------------------------------
# Step-0 full energy/virial consistency
# ---------------------------------------------------------------------------


@pytest.mark.dd12
class TestStep0FullEnergy:
    """All energy terms at step 0 must match across configurations."""

    def _step0(self, result: MDRunResult):
        f = result.mdlog.frames[0]
        assert f.step == 0
        return f

    def test_potential_matches(self, serial_run, dd12_run):
        s = self._step0(serial_run)
        d = self._step0(dd12_run)
        assert d.potential == pytest.approx(s.potential, abs=0.01)

    def test_kinetic_matches(self, serial_run, dd12_run):
        s = self._step0(serial_run)
        d = self._step0(dd12_run)
        assert d.kinetic == pytest.approx(s.kinetic, abs=0.01)

    def test_total_matches(self, serial_run, dd12_run):
        s = self._step0(serial_run)
        d = self._step0(dd12_run)
        assert d.total == pytest.approx(s.total, abs=0.01)

    def test_pressure_matches(self, serial_run, dd12_run):
        s = self._step0(serial_run)
        d = self._step0(dd12_run)
        assert d.pressure == pytest.approx(s.pressure, abs=1.0)

    def test_temperature_matches(self, serial_run, dd12_run):
        s = self._step0(serial_run)
        d = self._step0(dd12_run)
        assert d.temperature == pytest.approx(s.temperature, abs=0.01)


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    """Energy drift should be reasonable (not catastrophic)."""

    def test_serial_drift(self, serial_run):
        drift = serial_run.mdlog.energy_drift
        assert drift is not None, "No energy drift reported"
        assert abs(drift) < 5.0, (
            f"Serial energy drift {drift:.3f} kJ/mol/ps/atom too large"
        )

    @pytest.mark.dd12
    def test_dd12_drift(self, dd12_run):
        drift = dd12_run.mdlog.energy_drift
        assert drift is not None, "No energy drift reported"
        assert abs(drift) < 5.0, (
            f"DD12 energy drift {drift:.3f} kJ/mol/ps/atom too large"
        )

    @pytest.mark.dd12
    def test_drift_similar_across_dd(self, serial_run, dd12_run):
        """DD should not make drift dramatically worse."""
        d_serial = serial_run.mdlog.energy_drift
        d_dd = dd12_run.mdlog.energy_drift
        if d_serial is not None and d_dd is not None:
            # DD drift should be within 5x of serial (generous margin)
            assert abs(d_dd) < 5 * abs(d_serial) + 1.0, (
                f"DD12 drift {d_dd:.3f} much worse than serial {d_serial:.3f}"
            )


# ---------------------------------------------------------------------------
# Debug log validation (MPI energy sum consistency)
# ---------------------------------------------------------------------------


@pytest.mark.dd12
class TestDebugLogs:
    """Validate per-rank debug logs for consistency."""

    def test_mpi_sum_consistent_across_ranks(self, dd12_run):
        """All ranks should report the same mpiSum energy at each step."""
        debug_files = sorted(dd12_run.workdir.glob("metatomic_debug_rank_*.log"))
        if not debug_files:
            pytest.skip("No debug logs found")

        all_entries = [parse_debug_log(f) for f in debug_files]

        # Check that all ranks have the same number of entries
        lengths = [len(e) for e in all_entries]
        assert len(set(lengths)) == 1, f"Ranks have different entry counts: {lengths}"

        # Check mpiSum is identical across ranks at each step
        for step_idx in range(lengths[0]):
            sums = [entries[step_idx].energy_mpi_sum for entries in all_entries]
            assert all(s == pytest.approx(sums[0], abs=1e-4) for s in sums), (
                f"Step {step_idx}: mpiSum varies across ranks: {sums}"
            )

    def test_home_atoms_sum_to_total(self, dd12_run):
        """Sum of numHomeMta across ranks should equal total MTA atoms."""
        debug_files = sorted(dd12_run.workdir.glob("metatomic_debug_rank_*.log"))
        if not debug_files:
            pytest.skip("No debug logs found")

        all_entries = [parse_debug_log(f) for f in debug_files]
        n_atoms = dd12_run.mdlog.n_atoms
        if n_atoms is None:
            pytest.skip("Could not parse atom count")

        # Check first step
        home_sum = sum(entries[0].num_home_mta for entries in all_entries)
        assert home_sum == n_atoms, (
            f"Sum of home atoms {home_sum} != total atoms {n_atoms}"
        )


# ---------------------------------------------------------------------------
# Timer validation
# ---------------------------------------------------------------------------


@pytest.mark.dd12
class TestTimerOutput:
    """Validate timer log output structure and performance."""

    def _avgs(self, result: MDRunResult) -> dict[str, float]:
        if not result.timers.averages:
            pytest.skip("No timer data (GMX_METATOMIC_TIMER not set?)")
        return result.timers.averages

    def test_timer_logs_exist(self, dd12_run):
        timer_files = list(dd12_run.workdir.glob("metatomic_timer_rank_*.log"))
        assert len(timer_files) == 12, (
            f"Expected 12 timer logs, found {len(timer_files)}"
        )

    def test_required_timers_present(self, dd12_run):
        avgs = self._avgs(dd12_run)
        for required in [
            "calculateForces",
            "tensorPrep",
            "buildNL",
            "forward",
            "backward",
        ]:
            assert required in avgs, f"Timer '{required}' not found in output"

    def test_forward_not_zero(self, dd12_run):
        avgs = self._avgs(dd12_run)
        assert avgs.get("forward", 0) > 0.01, "forward timer suspiciously small"

    def test_buildNL_not_dominant(self, dd12_run):
        """buildNL should not dominate total time (regression for check_consistency)."""
        avgs = self._avgs(dd12_run)
        build = avgs.get("buildNL", 0)
        total = avgs.get("calculateForces", 0)
        if total > 0:
            ratio = build / total
            assert ratio < 0.3, (
                f"buildNL takes {ratio:.0%} of calculateForces "
                f"({build:.3f}ms / {total:.3f}ms) — "
                f"check_consistency may still be enabled"
            )

    def test_registerAutograd_not_dominant(self, dd12_run):
        """registerAutograd should be fast when check_consistency=false."""
        avgs = self._avgs(dd12_run)
        reg = avgs.get("registerAutograd", 0)
        total = avgs.get("calculateForces", 0)
        if total > 0 and reg > 0:
            ratio = reg / total
            assert ratio < 0.2, (
                f"registerAutograd takes {ratio:.0%} of calculateForces "
                f"({reg:.3f}ms / {total:.3f}ms) — "
                f"check_consistency may still be enabled"
            )

    def test_forward_dominates(self, dd12_run):
        """Model forward pass should be the largest single phase."""
        avgs = self._avgs(dd12_run)
        fwd = avgs.get("forward", 0)
        build = avgs.get("buildNL", 0)
        prep = avgs.get("tensorPrep", 0)
        if fwd > 0:
            assert fwd > build, (
                f"forward ({fwd:.3f}ms) should exceed buildNL ({build:.3f}ms)"
            )
            assert fwd > prep, (
                f"forward ({fwd:.3f}ms) should exceed tensorPrep ({prep:.3f}ms)"
            )
