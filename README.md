# Balancing Accuracy and Efficiency: Evaluating Encoder- and Decoder-Based Models for Word Sense Disambiguation and Regular Polysemy Detection

This repository contains the code accompanying the manuscript *"Balancing
Accuracy and Efficiency: Evaluating Encoder- and Decoder-Based Models for Word
Sense Disambiguation and Regular Polysemy Detection"* by Pin-Er Chen, Da-Chen
Lian, and Shu-Kai Hsieh (Graduate Institute of Linguistics, National Taiwan
University), currently under review at *Natural Language Processing*
(Cambridge University Press).

> **Status:** the manuscript is under peer review. Citation details below
> (volume, pages, DOI) will be updated once the article is accepted and
> assigned its final bibliographic information.

## Abstract

This study investigates the nuanced challenges of fine-grained Word Sense
Disambiguation (WSD) tasks with Regular Polysemy Detection (RPD) of the Named
Entity, focusing on evaluating the trade-offs between encoder and
decoder-based model performance and computational efficiency. The datasets,
including Chinese Wordnet 2.0 (CWN) as sense inventory, the Social Media
Corpus (PTT) for user-generated content, and the Academia Sinica Balanced
Corpus (ASBC) for formal linguistic data, were chosen to provide a diverse
and representative framework for evaluating both common nouns and proper
nouns with regular polysemy in Taiwan Mandarin. This analysis evaluated ten
encoder- and decoder-based models, assessing their performance on two tasks.
The encoder-based models demonstrate comparable accuracy to the
decoder-based models on WSD tasks (77.5 per cent vs. 78.5 per cent), and
similarly strong performance in RPD tasks (84.2 per cent vs. 83.8 per cent).
On a large-scale all-words word sense disambiguation task, the encoder model
not only outperformed the decoder model but also generated substantially
lower carbon emissions—an eight-fold reduction. These differences
underscore the trade-offs between model architecture and task-specific
performance, highlighting the necessity for balancing performance and energy
efficiency in the design and application of language models, advocating for
sustainable and eco-friendly practices in NLP development.

## Citation

> The manuscript is under review; once accepted, this section will be updated
> with the final journal citation (volume, pages, DOI).

Plain text (provisional):

> Chen, P.-E., Lian, D.-C., & Hsieh, S.-K. Balancing Accuracy and Efficiency:
> Evaluating Encoder- and Decoder-Based Models for Word Sense Disambiguation
> and Regular Polysemy Detection. *Natural Language Processing*. Cambridge
> University Press. Manuscript under review.

BibTeX (provisional):

```bibtex
@unpublished{chen_balancing_underreview,
  title   = {Balancing Accuracy and Efficiency: Evaluating Encoder- and Decoder-Based Models for Word Sense Disambiguation and Regular Polysemy Detection},
  author  = {Chen, Pin-Er and Lian, Da-Chen and Hsieh, Shu-Kai},
  note    = {Manuscript submitted to Natural Language Processing (Cambridge University Press); under review}
}
```

See `CITATION.cff` for the machine-readable form.

## Repository contents

| Path | Purpose |
|---|---|
| `src/dotted_wsd/` | Library code: dataset loaders, training, evaluation, ASBC large-scale tagging |
| `src/dotted_wsd/asbc_eval/` | §5 large-scale ASBC tagging pipeline (see its own `README.md`) |
| `scripts/` | Shell entry points for training, held-out evaluation, and ASBC large-scale tagging |
| `tokenizers/{configs,customized,default}/` | Per-base tokenizer YAMLs and their customized / reference JSONs |
| `tokenizers/customize.py` | Reproduces `tokenizers/customized/*.json` from `tokenizers/default/*.json` |
| `carbon_emissions/` | Reproduction package for the paper's emissions numbers (see its own `README.md`) |
| `tests/` | Pytest suite: imports, CLI `--help`, `carbon_emissions` reproducibility, `deduplicate_instances` round-trip, `customize.py` byte-equality |
| `pyproject.toml` / `uv.lock` | `uv`-managed Python environment |

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
# Install uv if you don't already have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the locked environment (creates .venv/, installs all pinned dependencies)
uv sync
```

Requires Python ≥ 3.10. PyTorch is pulled from the CUDA 12.1 wheel index (see
`[tool.uv.sources]` in `pyproject.toml`); machines without a CUDA 12.1-compatible
GPU may need to adjust the index before running `uv sync`.

`uv.lock` is committed and pins the dependency versions this release was
tested against. Note that the lock has been refreshed since the original
internal training environment (e.g. to add `pandas`/`numpy`/`tqdm` as
direct deps and to bump a yanked transitive `protobuf` version), so it
is not byte-identical to the snapshot that produced the paper's released
weights, but resolved versions of the load-bearing libraries
(`torch`, `transformers`, `peft`, `accelerate`, etc.) are unchanged.

## Data

**No data ships with this repository.** You must obtain and place the
following files under a top-level `data/` directory yourself. The
`data/.gitkeep` placeholder is committed only so the directory exists
on a fresh clone.

```text
data/
├── WSD_merge_train_v2.csv              # WSD training split (context-gloss pairs, "Reproducing training")
├── WSD_merge_test_v2.csv               # WSD held-out test split (used by training and evaluation)
├── RP_train.csv                         # RPD training split
├── RP_valid.csv                         # RPD held-out test split
├── wsd_examples.csv                     # By-example WSD eval set (used by `dwsd_eval`)
├── glossdict.json                       # CWN gloss dictionary
└── dt-asbc/                             # Raw ASBC tagged .txt files (input to §5 preprocessing)
    ├── asbc_dotted_tagged_000-of-140.txt
    ├── asbc_dotted_tagged_001-of-140.txt
    └── ...                              # 140 files in total
```

Two further files are *produced* by the preprocessing pipeline (see the
[Preprocessing](#preprocessing) section), not supplied:

- `data/dt_asbc_dataset/*.csv` — per-file WSD instance CSVs from
  `process_into_instances`.
- `data/asbc-deduplicated-instances.csv` — the deduplicated dataset,
  human-readable.
- `data/asbc-deduplicated-instances.feather` — the same dataset as
  zstd-compressed feather; this is what `scripts/run_asbc_eval.sh` reads.

The data files listed above are **not produced by this repo** and are not
shipped with it.

The library reads these paths via `dotted_wsd.settings.DATA_DIR`, which
defaults to `<repo>/data/`.

## Preprocessing

The 140 raw ASBC tagged `.txt` files are turned into the deduplicated
feather dataset that `scripts/run_asbc_eval.sh` consumes via two scripts.
Set up the input directory once, then run the two commands.

1. **Place raw files** under `data/dt-asbc/` — see the
   [Data](#data) section.

2. **Per-file WSD instance extraction** (looks up candidate senses in CWN):

   ```bash
   uv run python -m dotted_wsd.asbc_eval.process_into_instances
   ```

   Outputs one `*-eval-prepared.csv` per input file under
   `data/dt_asbc_dataset/` (the directory is created automatically).
   Add `--debug` to process only the first two input files for a smoke
   test.

3. **Cross-file deduplication** of `(test_sentence, test_word)` pairs:

   ```bash
   uv run python -m dotted_wsd.asbc_eval.deduplicate_instances
   ```

   Outputs three files at the top of `data/`:

   - `asbc-deduplicated-instances.csv` — human-readable
   - `asbc-deduplicated-instances.feather` — zstd-compressed; what
     `scripts/run_asbc_eval.sh` reads
   - `test_sentence_to_example_ids.json` — sidecar mapping each
     deduplicated `test_sentence` back to every `example_id` that was
     collapsed into it

   Pass `--save-dir <path>` to redirect.

The decoder-based bases (Llama-3.2-3B, Gemma-2-2b, SmolLM-180M) use
customized tokenizer JSONs vendored under `tokenizers/customized/`; the
unmodified reference JSONs are at `tokenizers/default/`. The customization
registers one or two special tokens with each tokenizer's Rust-level
`post_processor` and rewrites the pair template; per-base details are
encoded as deterministic patch functions in
[`tokenizers/customize.py`](tokenizers/customize.py):

```bash
uv run python tokenizers/customize.py            # rewrite tokenizers/customized/*.json
uv run python tokenizers/customize.py --check    # diff against vendored files; exit 1 on drift
```

Training and evaluation scripts read the customized files via the per-base
YAMLs in `tokenizers/configs/`. The shipped JSONs are sufficient for
reproduction — no extra step. Run `customize.py` only if you want to adapt
the customization to a new base model (use one of the existing patch
functions as a template).

The training data CSVs are not produced by this repo and are not openly
available — see the [Data](#data) section.

## Fine-tuned models on the Hugging Face Hub

13 `lopentu/*-DottedWSD` checkpoints are released under the
[`lopentu`](https://huggingface.co/lopentu) organization. They are
**currently private** while the manuscript is under review and will be made
public once the article is accepted. The paper itself reports on a subset
of 10 of these.

- `lopentu/google-bert-bert-base-chinese-DottedWSD`
- `lopentu/ckiplab-bert-base-chinese-DottedWSD`
- `lopentu/yentinglin-bert-base-zhtw-DottedWSD`
- `lopentu/IDEA-CCNL-Erlangshen-DeBERTa-v2-97M-Chinese-DottedWSD`
- `lopentu/MoritzLaurer-mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-DottedWSD`
- `lopentu/MoritzLaurer-mDeBERTa-v3-base-mnli-xnli-DottedWSD`
- `lopentu/microsoft-mdeberta-v3-base-DottedWSD`
- `lopentu/microsoft-deberta-v3-small-DottedWSD`
- `lopentu/microsoft-deberta-v3-base-DottedWSD`
- `lopentu/microsoft-deberta-v3-large-DottedWSD`
- `lopentu/SmolLM-Chinese-180M-DottedWSD`
- `lopentu/gemma-2-2b-DottedWSD`
- `lopentu/meta-llama-Llama-3.2-3B-DottedWSD`

## Reproducing training

The training entry point is `dotted_wsd.train.hf_trainer`. To reproduce the
full grid of fine-tunes from the paper:

```bash
# Edit scripts/run_hf_trainer.sh to comment out any configs you don't want
bash scripts/run_hf_trainer.sh
```

By default:

- Training metrics are not reported anywhere (`--report-to none`).
- The trained model is **not** pushed to the Hugging Face Hub (`--no-push-to-hub`).

To log to Weights & Biases, pass `--report-to wandb` and (optionally) override
the destination via env vars:

```bash
WANDB_ENTITY=your-entity WANDB_PROJECT=your-project \
  uv run python -m dotted_wsd.train.hf_trainer <model_id> <batch_size> --report-to wandb
```

To push the trained model to the Hub, pass `--push-to-hub` and a
`--hub-model-id` you can write to:

```bash
uv run python -m dotted_wsd.train.hf_trainer <model_id> <batch_size> \
  --push-to-hub --hub-model-id your-namespace/your-model-name
```

If `--hub-model-id` is omitted, it defaults to
`lopentu/{model_id}-DottedWSD{suffix}` — only useful if you have write access
to the `lopentu` org.

## Reproducing evaluation

```bash
bash scripts/run_eval.sh
```

Reads each released `lopentu/*-DottedWSD` model from the Hugging Face Hub
and writes per-model evaluation results under
`data/eval_results/<model_name>/` (gitignored). The output for each model
is two pickle files (`*_wsd_eval.pkl` and `*_rp_eval.pkl`) plus a few
PNG plots from the by-example analysis. The WSD pickle contains
`{metadata, by_example, by_instance}` (per-example/instance prediction
DataFrames + accuracy values); the RP pickle contains
`{hint, nohint}`, each holding a sklearn-style classification report
DataFrame and a per-row prediction list.

## Reproducing the §5 large-scale ASBC tagging experiment

The §5 experiment runs each fine-tuned model over the full ASBC corpus
(~32.7M instances). It assumes you've already produced
`data/asbc-deduplicated-instances.feather` via the
[ASBC corpus preparation](#2-asbc-corpus-preparation-5-input)
preprocessing pipeline.

```bash
bash scripts/run_asbc_eval.sh
```

Tagging outputs land under `data/asbc_eval_results/<model_name>/`. The
shipped `run_asbc_eval.sh` includes `--debug` for safety; **remove the
`--debug` flag for a full production run.** When using `--debug` with
the default `--preprocess-workers` (≈ `cpu_count()`), the tiny debug
dataset can trigger `BrokenProcessPool`; pass `--preprocess-workers 8`
in that case.

For the per-model energy and CO₂e numbers cited in the paper's abstract,
see [`carbon_emissions/`](carbon_emissions/) — input CSV plus three
stdlib-only Python scripts that reproduce the totals, run the
sensitivity analysis, and verify each headline number.

See [`src/dotted_wsd/asbc_eval/README.md`](src/dotted_wsd/asbc_eval/README.md)
for a tighter end-to-end recipe.

## License

This repository's source code is released under the MIT License — see
[`LICENSE`](LICENSE).

## Acknowledgements / third-party model licenses

The MIT license in this repository covers the **source code only**. Each
released `lopentu/*-DottedWSD` checkpoint is a fine-tuned derivative of an
upstream pretrained model, and is bound by the upstream model's license —
**which in several cases is more restrictive than MIT**. Verify the
upstream license on every model card you build on.

The mapping below was read directly from each base model's `card_data.license`
field on the Hugging Face Hub.

| Base model | Upstream license | DottedWSD checkpoint |
|---|---|---|
| `meta-llama/Llama-3.2-3B` | **Llama 3.2 Community License** (custom, non-OSI) | `lopentu/meta-llama-Llama-3.2-3B-DottedWSD` |
| `google/gemma-2-2b` | **Gemma Terms of Use** (custom, non-OSI) | `lopentu/gemma-2-2b-DottedWSD` |
| `Mxode/SmolLM-Chinese-180M` | **GPL-3.0** (copyleft) | `lopentu/SmolLM-Chinese-180M-DottedWSD` |
| `ckiplab/bert-base-chinese` | **GPL-3.0** (copyleft) | `lopentu/ckiplab-bert-base-chinese-DottedWSD` |
| `yentinglin/bert-base-zhtw` | **CC-BY-NC-SA-4.0** (non-commercial) | `lopentu/yentinglin-bert-base-zhtw-DottedWSD` |
| `microsoft/deberta-v3-{small,base,large}` | MIT | `lopentu/microsoft-deberta-v3-{small,base,large}-DottedWSD` |
| `microsoft/mdeberta-v3-base` | MIT | `lopentu/microsoft-mdeberta-v3-base-DottedWSD` |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | MIT | `lopentu/MoritzLaurer-mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-DottedWSD` |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | MIT | `lopentu/MoritzLaurer-mDeBERTa-v3-base-mnli-xnli-DottedWSD` |
| `IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese` | Apache-2.0 | `lopentu/IDEA-CCNL-Erlangshen-DeBERTa-v2-97M-Chinese-DottedWSD` |
| `google-bert/bert-base-chinese` | Apache-2.0 | `lopentu/google-bert-bert-base-chinese-DottedWSD` |
| `WENGSYX/Deberta-Chinese-Large` | (license field is unset on the upstream model card; treat as unknown until clarified) | (no released checkpoint) |

Practical implications:

- **Don't redistribute the `yentinglin-bert-base-zhtw-DottedWSD` checkpoint
  for commercial use** — `CC-BY-NC-SA-4.0` forbids it.
- The `SmolLM-Chinese-180M-DottedWSD` and `ckiplab-bert-base-chinese-DottedWSD`
  checkpoints inherit GPL-3.0 from the upstream weights; redistributing
  them or works that link them likely requires GPL-3.0 compliance.
- The `Llama` and `Gemma` checkpoints are subject to Meta's and Google's
  custom community/terms-of-use licenses, which include acceptable-use
  policies.
- The remaining checkpoints are under permissive (MIT or Apache-2.0)
  upstream licenses, so MIT redistribution of the fine-tuned weights is
  generally safe — but always re-check the model card before assuming.
