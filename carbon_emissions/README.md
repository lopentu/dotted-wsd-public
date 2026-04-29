# Carbon emissions analysis

Reproducible per-run carbon-emissions estimates for the 13 fine-tuning runs
backing the paper. All 13 runs were performed in November 2024 across
on-premises (National Taiwan University) and cloud (RunPod) infrastructure.

## Files

| File | Purpose |
|---|---|
| `calculate_carbon.py` | Print every calculated value (totals, per-infra split, sensitivity, per-run rows) |
| `verify_report.py` | Re-derive each headline number from the CSV and assert it; exit 0 = all pass |
| `model_breakdown.py` | Per-model and encoder/decoder grouped tables |
| `dotted_wsd_hardware_duration.csv` | Input data: 13 W&B runs exported with hardware/runtime/energy fields |

## Usage

```bash
python3 calculate_carbon.py dotted_wsd_hardware_duration.csv
python3 verify_report.py    dotted_wsd_hardware_duration.csv
python3 model_breakdown.py  dotted_wsd_hardware_duration.csv
```

No dependencies beyond Python 3 and the standard library.

## Method

```
CO₂e (kg) = E_gpu (kWh) × f_sys × PUE × CI
```

| Term | Description |
|---|---|
| E_gpu | GPU board energy via NVML (`avg_gpu_power_watts × runtime_hours / 1000`) |
| f_sys | System-overhead multiplier for CPU, RAM, storage, fans, VRMs |
| PUE | Power Usage Effectiveness (facility cooling, UPS, power distribution) |
| CI | Grid carbon intensity (kgCO₂eq/kWh) |

The framework follows Patterson et al. (2022). Because we measure only GPU
board power (NVML) rather than total server power, we add an explicit
system-overhead factor `f_sys` to recover the full server draw, following
Luccioni et al. (2023), who found GPU power accounted for ~54% of total node
power on the Jean Zay supercomputer. Bouza et al. (2023) review the
comparable approaches.

### Parameters

| Parameter | On-prem | Cloud (RunPod) | Source |
|---|---|---|---|
| f_sys | 1.5 | 1.5 | Luccioni et al. (2023); conservative vs. BLOOM's 1.85 |
| PUE | 1.2 | 1.3 | University server room / RunPod does not publish |
| CI (kgCO₂eq/kWh) | 0.474 | 0.475 | Taiwan Energy Administration 2024 / conservative upper bound |

Taiwan CI source: Ministry of Economic Affairs, Energy Administration (能源署),
*Energy Statistics Handbook 2024* (能源統計手冊), p. 15: 電力排碳係數 = 0.474 kgCO₂eq/kWh
for 2024.

### Data sources

- **Energy.** Weights & Biases run logs. Power values sampled from the NVML
  hardware counter (`nvmlDeviceGetPowerUsage`), not estimated from TDP.
- **GPUs.** NVIDIA RTX A5000 (Ampere, 230W TDP) and NVIDIA GeForce RTX 4090
  (Ada Lovelace, 450W TDP).
- **Infrastructure.** On-prem: NTU server (`unicorn`). Cloud: two RunPod
  containers.

## Results

### By infrastructure

| Infrastructure | GPU | Runs | GPU energy | Runtime | CO₂e |
|---|---|---|---|---|---|
| On-prem (NTU) | RTX A5000 | 3 | 6.46 kWh | 28.7 h | 5.5 kg |
| Cloud (RunPod) | RTX 4090 | 10 | 9.34 kWh | 29.5 h | 7.0–8.6 kg |
| **Total** | | **13** | **15.80 kWh** | **58.2 h** | **12.5–14.2 kg** |

The cloud range reflects CI uncertainty (0.386–0.475 kgCO₂/kWh). On-prem is a
point estimate.

### Sensitivity to overhead assumptions

| Scenario | Effective energy | CO₂e | Notes |
|---|---|---|---|
| GPU-only, no overhead | 15.8 kWh | 7.5 kg | Lower bound; ignores CPU, RAM, cooling |
| **f_sys=1.5, PUE per infrastructure** | **29.8 kWh** | **14.2 kg** | **Central estimate** |
| f_sys=1.85, PUE per infrastructure | 36.8 kWh | 17.5 kg | Upper bound; HPC-cluster-scale overhead |

All rows use infrastructure-specific PUE (1.2 on-prem, 1.3 cloud) and CI
(0.474 on-prem, 0.475 cloud). The "GPU-only" row omits f_sys and PUE but
retains the per-infrastructure CI.

### By architecture type

| Type | Runs | GPU energy | Runtime | CO₂e | % energy | % CO₂e |
|---|---|---|---|---|---|---|
| Decoder | 3 | 10.07 kWh | 38.4 h | 8,908 g | 63.8% | 62.9% |
| Encoder | 10 | 5.72 kWh | 19.8 h | 5,252 g | 36.2% | 37.1% |
| **Total** | **13** | **15.80 kWh** | **58.2 h** | **14,160 g** | **100%** | **100%** |

The three decoder runs (Llama-3.2-3B, Gemma-2-2b, SmolLM-180M) produced 1.7×
the emissions of all ten encoder runs combined. The two largest (Llama-3.2-3B
and Gemma-2-2b) accounted for 54.4% of the total. For a per-model breakdown,
run `model_breakdown.py`.

## Limitations

- **Cloud region unknown.** RunPod containers were identified only by
  Docker container IDs; the datacenter location could not be recovered.
  Cloud runs account for 59.1% of total GPU energy, so the cloud CI
  assumption is consequential. If cloud CI were 0.04 (e.g. Quebec hydro),
  total emissions would drop to ~6 kg; if 0.509 (coal-heavy grid), total
  would be ~15 kg, a range of ~9 kg.
- **PUE not disclosed by RunPod.** PUE=1.3 is based on colocation industry
  averages and should be treated as approximate.
- **System overhead not directly measured.** The 1.5× `f_sys` is based on
  published single-GPU server measurements. Actual overhead depends on
  CPU load, storage I/O, and chassis design, none of which were logged.
- **NVML power sampling.** W&B polls `nvmlDeviceGetPowerUsage` at intervals
  rather than integrating the hardware energy counter continuously. This
  introduces minor integration error, estimated at <5% for runs longer than
  a few minutes; all 13 runs exceed this threshold.
- **Embodied carbon excluded.** Manufacturing emissions for GPUs, servers,
  and datacenter infrastructure are not included. For fine-tuning runs at
  this scale, the embodied fraction is expected to be small but nonzero.

## References

- Patterson, D., Gonzalez, J., Hölzle, U., Le, Q., Liang, C., Munguia, L.-M.,
  Rothchild, D., So, D. R., Texier, M., & Dean, J. (2022). The carbon
  footprint of machine learning training will plateau, then shrink.
  *Computer*, 55(7), 18–28.
- Luccioni, A. S., Viguier, S., & Ligozat, A.-L. (2023). Estimating the
  carbon footprint of BLOOM, a 176B parameter language model. *JMLR*,
  24(253), 1–15.
- Energy Administration (能源署), Ministry of Economic Affairs. (2024).
  *Energy Statistics Handbook 2024* (能源統計手冊).
- Bouza, L., Bugeau, A., & Lannelongue, L. (2023). How to estimate carbon
  footprint when training deep learning models? A guide and review.
  *Environmental Research Communications*, 5(11), 115014.
