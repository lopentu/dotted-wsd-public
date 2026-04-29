# ASBC Evaluation

Pipeline that turns the raw Academia Sinica Balanced Corpus (ASBC) tagged
files into the deduplicated dataset used in §5 of the paper.

## Inputs

Place the raw ASBC tagged files (`asbc_dotted_tagged_NNN-of-140.txt`,
N=000..139) under `<repo>/data/dt-asbc/`.

## Steps

1. **Per-file WSD instance generation** — `process_into_instances.py`

   ```bash
   uv run python -m dotted_wsd.asbc_eval.process_into_instances
   ```

   Reads each `data/dt-asbc/*.txt` file, looks up candidate senses in CWN,
   and writes one `*-eval-prepared.csv` per input under
   `data/dt_asbc_dataset/`. Add `--debug` to only process the first two
   files (useful for smoke-testing the pipeline).

2. **Cross-file deduplication** — `deduplicate_instances.py`

   ```bash
   uv run python -m dotted_wsd.asbc_eval.deduplicate_instances
   ```

   Iterates `data/dt_asbc_dataset/*.csv`, groups by `test_sentence` (which
   already encodes the `<test_word>` markup), keeps one `example_id` per
   unique sentence, and concatenates the survivors. Writes:

   - `data/asbc-deduplicated-instances.csv` — human-readable row-per-instance
   - `data/asbc-deduplicated-instances.feather` — zstd-compressed; the input
     format the tagging script expects
   - `data/test_sentence_to_example_ids.json` — sidecar mapping each
     deduplicated `test_sentence` back to every `example_id` that was
     collapsed into it

   Override the destination with `--save-dir <path>`.

3. **Tag with each model** — `scripts/run_asbc_eval.sh`. Outputs land
   under `data/asbc_eval_results/` (gitignored).
