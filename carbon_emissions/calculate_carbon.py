#!/usr/bin/env python3
"""
Carbon emissions calculations for the 13 fine-tuning runs.

Reads `dotted_wsd_hardware_duration.csv` and prints totals, the
by-infrastructure split, sensitivity to overhead assumptions, an
energy-derivation sanity check, and per-run rows. See `README.md`
for the formula and parameter sources.
"""

import csv
import sys

# ── Parameters ──────────────────────────────────────────────
FSYS = 1.5  # system overhead multiplier
PUE_ONPREM = 1.2  # on-prem PUE (university server room)
PUE_CLOUD = 1.3  # cloud PUE (RunPod, assumed)
CI_ONPREM = 0.474  # Taiwan 2024, Energy Administration (能源署)
CI_CLOUD_LOW = 0.386  # US national average
CI_CLOUD_HIGH = 0.475  # conservative upper bound

# ── Load data ───────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "dotted_wsd_hardware_duration.csv"

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

op = [r for r in rows if r["infrastructure"] == "on-prem"]
cl = [r for r in rows if r["infrastructure"] == "cloud"]

op_e = sum(float(r["energy_kwh"]) for r in op)
cl_e = sum(float(r["energy_kwh"]) for r in cl)
op_h = sum(float(r["runtime_hours"]) for r in op)
cl_h = sum(float(r["runtime_hours"]) for r in cl)
total_e = op_e + cl_e
total_h = op_h + cl_h

# ── §4 Results: By infrastructure ───────────────────────────
op_co2 = op_e * FSYS * PUE_ONPREM * CI_ONPREM
cl_co2_low = cl_e * FSYS * PUE_CLOUD * CI_CLOUD_LOW
cl_co2_high = cl_e * FSYS * PUE_CLOUD * CI_CLOUD_HIGH

print("=" * 60)
print("§4 Results: By infrastructure")
print("=" * 60)
print(f"On-prem:  {len(op):2d} runs | {op_e:.2f} kWh | {op_h:.1f} h | {op_co2:.1f} kg CO2e")
print(
    f"Cloud:    {len(cl):2d} runs | {cl_e:.2f} kWh | {cl_h:.1f} h | {cl_co2_low:.1f}–{cl_co2_high:.1f} kg CO2e"
)
print(
    f"Total:    {len(rows):2d} runs | {total_e:.2f} kWh | {total_h:.1f} h | {op_co2 + cl_co2_low:.1f}–{op_co2 + cl_co2_high:.1f} kg CO2e"
)
print(f"\nCloud energy fraction: {cl_e / total_e * 100:.1f}%")

# ── §4 Results: Sensitivity ─────────────────────────────────
print(f"\n{'=' * 60}")
print("§4 Results: Sensitivity to overhead assumptions")
print("=" * 60)
print(f"{'Scenario':<45} {'Eff. energy':>12} {'CO2e':>10}")
print("-" * 67)

for label, fsys in [
    ("GPU-only, no overhead", 1.0),
    ("f_sys=1.5, PUE per infrastructure", 1.5),
    ("f_sys=1.85, PUE per infrastructure", 1.85),
]:
    if fsys == 1.0:
        op_eff, cl_eff = op_e, cl_e
    else:
        op_eff = op_e * fsys * PUE_ONPREM
        cl_eff = cl_e * fsys * PUE_CLOUD
    total_eff = op_eff + cl_eff
    co2 = op_eff * CI_ONPREM + cl_eff * CI_CLOUD_HIGH
    print(f"{label:<45} {total_eff:>10.1f} kWh {co2:>8.1f} kg")

# ── §5 Limitations: Cloud CI extremes ───────────────────────
print(f"\n{'=' * 60}")
print("§5 Limitations: Cloud CI sensitivity")
print("=" * 60)
for ci_label, ci in [
    ("0.04 (hydro/nuclear)", 0.04),
    ("0.386 (US average)", 0.386),
    ("0.474 (Taiwan 2024)", 0.474),
    ("0.475 (conservative)", 0.475),
    ("0.509 (coal-heavy)", 0.509),
]:
    cl_co2 = cl_e * FSYS * PUE_CLOUD * ci
    print(f"  Cloud CI = {ci_label:<25} total = {op_co2 + cl_co2:.1f} kg")

full_range = cl_e * FSYS * PUE_CLOUD * (0.509 - 0.04)
print(f"  Full range width: ~{full_range:.0f} kg")

# ── Energy derivation check ─────────────────────────────────
print(f"\n{'=' * 60}")
print("Energy derivation check (avg_gpu_power_watts × runtime_hours / 1000)")
print("=" * 60)
mismatches = 0
for r in rows:
    watts = float(r["avg_gpu_power_watts"])
    hours = float(r["runtime_hours"])
    expected = watts * hours / 1000
    actual = float(r["energy_kwh"])
    diff_pct = abs(expected - actual) / actual * 100 if actual > 0 else 0
    if diff_pct > 1:
        print(
            f"  MISMATCH: {r['run_name']} logged={actual:.4f} computed={expected:.4f} diff={diff_pct:.1f}%"
        )
        mismatches += 1
if mismatches == 0:
    print("  All values match within 1%. OK.")

# ── Per-run detail ──────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Per-run detail")
print("=" * 60)
print(f"{'Run name':<50} {'Infra':<8} {'GPU':<25} {'kWh':>7} {'Hours':>7} {'CO2e(g)':>8}")
print("-" * 108)
for r in sorted(rows, key=lambda x: float(x["energy_kwh"]), reverse=True):
    e = float(r["energy_kwh"])
    infra = r["infrastructure"]
    co2 = (
        e * FSYS * PUE_ONPREM * CI_ONPREM
        if infra == "on-prem"
        else e * FSYS * PUE_CLOUD * CI_CLOUD_HIGH
    )
    print(
        f"{r['run_name']:<50} {infra:<8} {r['gpu']:<25} {e:>7.3f} {float(r['runtime_hours']):>7.2f} {co2 * 1000:>8.0f}"
    )

# ── Host/GPU summary ────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Host and GPU summary")
print("=" * 60)
hosts = {}
for r in rows:
    h = r["host"]
    if h not in hosts:
        hosts[h] = {"gpu": set(), "infra": r["infrastructure"], "runs": 0}
    hosts[h]["gpu"].add(r["gpu"])
    hosts[h]["runs"] += 1
for h, info in sorted(hosts.items()):
    print(f"  {h}: {', '.join(info['gpu'])} ({info['infra']}, {info['runs']} runs)")
