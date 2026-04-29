#!/usr/bin/env python3
"""
Per-model and encoder/decoder grouped carbon-emissions tables.

Usage: python3 model_breakdown.py [path_to_csv]
"""

import csv
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "dotted_wsd_hardware_duration.csv"

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

# ── Parameters ──────────────────────────────────────────────
FSYS = 1.5
PUE_OP, PUE_CL = 1.2, 1.3
CI_OP, CI_CL = 0.474, 0.475

DECODERS = {
    "Mxode-SmolLM-Chinese-180M",
    "google-gemma-2-2b",
    "meta-llama-Llama-3.2-3B",
}


def co2_kg(row):
    e = float(row["energy_kwh"])
    if row["infrastructure"] == "on-prem":
        return e * FSYS * PUE_OP * CI_OP
    else:
        return e * FSYS * PUE_CL * CI_CL


# ── Annotate rows ───────────────────────────────────────────
for r in rows:
    r["_energy"] = float(r["energy_kwh"])
    r["_hours"] = float(r["runtime_hours"])
    r["_co2_kg"] = co2_kg(r)
    r["_type"] = "Decoder" if r["run_name"] in DECODERS else "Encoder"

total_co2 = sum(r["_co2_kg"] for r in rows)

# ── Table 1: Per-model breakdown ────────────────────────────
print("Table: Per-model carbon emissions (sorted by CO₂e, descending)")
print()
header = f"{'Model':<55} {'Type':<8} {'Infra':<8} {'kWh':>7} {'Hours':>7} {'CO₂e (g)':>9} {'%':>6}"
print(header)
print("-" * len(header))

for r in sorted(rows, key=lambda x: x["_co2_kg"], reverse=True):
    pct = r["_co2_kg"] / total_co2 * 100
    print(
        f"{r['run_name']:<55} {r['_type']:<8} {r['infrastructure']:<8} "
        f"{r['_energy']:>7.3f} {r['_hours']:>7.2f} {r['_co2_kg'] * 1000:>9.0f} {pct:>5.1f}%"
    )

print("-" * len(header))
print(
    f"{'Total':<55} {'':8} {'':8} "
    f"{sum(r['_energy'] for r in rows):>7.2f} "
    f"{sum(r['_hours'] for r in rows):>7.1f} "
    f"{total_co2 * 1000:>9.0f} {'100.0%':>6}"
)

# ── Table 2: Grouped by architecture type ───────────────────
print()
print()
print("Table: Emissions grouped by architecture type")
print()
header2 = f"{'Type':<10} {'Runs':>5} {'kWh':>8} {'Hours':>8} {'CO₂e (g)':>10} {'% energy':>9} {'% CO₂e':>8}"
print(header2)
print("-" * len(header2))

for label in ["Decoder", "Encoder"]:
    group = [r for r in rows if r["_type"] == label]
    e = sum(r["_energy"] for r in group)
    h = sum(r["_hours"] for r in group)
    co2 = sum(r["_co2_kg"] for r in group)
    total_e = sum(r["_energy"] for r in rows)
    print(
        f"{label:<10} {len(group):>5} {e:>8.2f} {h:>8.1f} "
        f"{co2 * 1000:>10.0f} {e / total_e * 100:>8.1f}% {co2 / total_co2 * 100:>7.1f}%"
    )

print("-" * len(header2))
total_e = sum(r["_energy"] for r in rows)
total_h = sum(r["_hours"] for r in rows)
print(
    f"{'Total':<10} {len(rows):>5} {total_e:>8.2f} {total_h:>8.1f} "
    f"{total_co2 * 1000:>10.0f} {'100.0%':>9} {'100.0%':>8}"
)

# ── Summary stats for main text ─────────────────────────────
dec = [r for r in rows if r["_type"] == "Decoder"]
enc = [r for r in rows if r["_type"] == "Encoder"]
dec_co2 = sum(r["_co2_kg"] for r in dec)
enc_co2 = sum(r["_co2_kg"] for r in enc)
dec_e = sum(r["_energy"] for r in dec)
enc_e = sum(r["_energy"] for r in enc)

top2 = sorted(rows, key=lambda x: x["_co2_kg"], reverse=True)[:2]
top2_co2 = sum(r["_co2_kg"] for r in top2)

print()
print()
print("Summary for main text:")
print(
    f"  Decoders: {len(dec)} runs, {dec_e:.2f} kWh, {dec_co2 * 1000:.0f} g CO2e ({dec_co2 / total_co2 * 100:.1f}%)"
)
print(
    f"  Encoders: {len(enc)} runs, {enc_e:.2f} kWh, {enc_co2 * 1000:.0f} g CO2e ({enc_co2 / total_co2 * 100:.1f}%)"
)
print(f"  Ratio: decoders produced {dec_co2 / enc_co2:.1f}x the emissions of encoders")
print(
    f"  Top 2 models ({', '.join(r['run_name'] for r in top2)}): {top2_co2 * 1000:.0f} g ({top2_co2 / total_co2 * 100:.1f}%)"
)
