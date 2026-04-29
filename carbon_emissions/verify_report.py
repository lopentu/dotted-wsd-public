#!/usr/bin/env python3
"""
Re-derive each headline number in `README.md` from the raw CSV and assert
it. Exit code 0 = all pass, 1 = at least one mismatch.

Usage: python3 verify_report.py [path_to_csv]
"""

import csv
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "dotted_wsd_hardware_duration.csv"

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

op = [r for r in rows if r["infrastructure"] == "on-prem"]
cl = [r for r in rows if r["infrastructure"] == "cloud"]

op_e = sum(float(r["energy_kwh"]) for r in op)
cl_e = sum(float(r["energy_kwh"]) for r in cl)
op_h = sum(float(r["runtime_hours"]) for r in op)
cl_h = sum(float(r["runtime_hours"]) for r in cl)

# Parameters
FSYS = 1.5
PUE_OP, PUE_CL = 1.2, 1.3
CI_OP = 0.474
CI_CL_LO, CI_CL_HI = 0.386, 0.475

failures: list[str] = []


def check(label, computed, expected, tol=0.05):
    ok = abs(computed - expected) <= tol
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(label)
    print(f"  [{status}] {label}: computed={computed:.2f}, expected={expected}, tol={tol}")


print("§4 By infrastructure")
check("Total runs", len(rows), 13, tol=0)
check("On-prem runs", len(op), 3, tol=0)
check("Cloud runs", len(cl), 10, tol=0)
check("On-prem energy (kWh)", op_e, 6.46)
check("Cloud energy (kWh)", cl_e, 9.34)
check("On-prem runtime (h)", op_h, 28.7, tol=0.1)
check("Cloud runtime (h)", cl_h, 29.5, tol=0.1)
check("Total energy (kWh)", op_e + cl_e, 15.80)
check("Total runtime (h)", op_h + cl_h, 58.2, tol=0.1)

op_co2 = op_e * FSYS * PUE_OP * CI_OP
cl_co2_lo = cl_e * FSYS * PUE_CL * CI_CL_LO
cl_co2_hi = cl_e * FSYS * PUE_CL * CI_CL_HI
check("On-prem CO2 (kg)", op_co2, 5.5, tol=0.1)
check("Cloud CO2 low (kg)", cl_co2_lo, 7.0, tol=0.1)
check("Cloud CO2 high (kg)", cl_co2_hi, 8.6, tol=0.1)
check("Total CO2 low (kg)", op_co2 + cl_co2_lo, 12.5, tol=0.1)
check("Total CO2 high (kg)", op_co2 + cl_co2_hi, 14.2, tol=0.1)

print("\n§4 Sensitivity")
for label, fsys, exp_eff, exp_co2 in [
    ("GPU-only", 1.0, 15.8, 7.5),
    ("1.5x", 1.5, 29.8, 14.2),
    ("1.85x", 1.85, 36.8, 17.5),
]:
    if fsys == 1.0:
        op_eff, cl_eff = op_e, cl_e
    else:
        op_eff = op_e * fsys * PUE_OP
        cl_eff = cl_e * fsys * PUE_CL
    total_eff = op_eff + cl_eff
    co2 = op_eff * CI_OP + cl_eff * CI_CL_HI
    check(f"{label} eff energy (kWh)", total_eff, exp_eff, tol=0.1)
    check(f"{label} CO2 (kg)", co2, exp_co2, tol=0.1)

print("\n§5 Cloud energy fraction")
check("Cloud fraction (%)", cl_e / (op_e + cl_e) * 100, 59.1, tol=0.1)

print("\n§5 Cloud CI extremes")
check("CI=0.04 total (kg)", op_co2 + cl_e * FSYS * PUE_CL * 0.04, 6.2, tol=0.5)
check("CI=0.509 total (kg)", op_co2 + cl_e * FSYS * PUE_CL * 0.509, 14.8, tol=0.5)

print("\nEnergy derivation check")
max_diff = 0
for r in rows:
    watts = float(r["avg_gpu_power_watts"])
    hours = float(r["runtime_hours"])
    expected = watts * hours / 1000
    actual = float(r["energy_kwh"])
    diff_pct = abs(expected - actual) / actual * 100 if actual > 0 else 0
    max_diff = max(max_diff, diff_pct)
check("Max derivation error (%)", max_diff, 0, tol=5.0)

print(f"\n{'=' * 40}")
if not failures:
    print("ALL CHECKS PASSED")
    sys.exit(0)
else:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
