# DataChain Qwen checkpoint demo

Compute one Qwen video embedding per window across all 12,219 input windows using 300 distributed workers, interrupt at source row 1,001, then resume using DataChain checkpoints.

## Files

- `qwen_checkpoint_demo.py`: the short job shown in the demo.
- `qwen_encoder.py`: native DataChain video decoding, DataModel schemas, and the pinned Qwen3-VL-Embedding-8B encoder.
- `requirements-cpu.txt`: CPU dependencies for Python 3.11.
- `LICENSE-QWEN.txt`: license for the upstream Qwen implementation.

## Run in Studio

1. Set the repository to `https://github.com/shcheklein/datachain-qwen-checkpoints@main`.
2. Paste `qwen_checkpoint_demo.py` into the job editor.
3. Use the GCP `default` cluster, Python 3.11, 300 distributed workers, and the contents of `requirements-cpu.txt`.
4. Run. The job raises `Demo interruption at source row 1,001` before embedding that row. Workers finish out of order, so this does not mean exactly 1,000 embeddings have completed. Use Studio's actual progress counters for the recording.
5. Remove the two-line `if row_id == 1001` failure block from the job editor, then click **Resume** on that failed job.

Keep the source version, row ID expression, ordering, output name, and return schema unchanged. The resumed job should reuse completed checkpoints, compute the remaining rows, and save all 12,219 rows. Model loading can happen again on each worker; checkpoints retain computed results, not an in-memory model. Buffered work that was not committed may repeat after a hard interruption.

The input is the full `@shcheklein.default.l2_egodex_qwen_7165f0d173@1.0.0` dataset. The earlier `basic_pick_place/244` filter selected only seven windows and is removed for this run. Source and output names are hard-coded directly in the pipeline. Each window uses 16 sampled frames and produces one normalized 4,096-dimensional embedding. The output is `@shcheklein.default.qwen_checkpoint_demo`.

`row_id` exposes the input dataset's existing `sys.id` to make the failure point readable. The input is ordered by that ID, but distributed completion order is not sequential. `sampling.index` cannot serve as a global counter because it restarts within each episode. The row ID condition only injects the failure; DataChain manages checkpoint reuse.

## Studio MCP

The connected `run_job` tool accepts a `repository` URL with an optional `@branch` or `@tag`. It also requires `query`, containing the Python source from `qwen_checkpoint_demo.py`. Its `query_name` is a display name, not a repository file selector. The encoder is imported from the cloned repository.

Pass the requirements file contents through `requirements`, set `python_version` to `3.11`, and use `workers: 300`. Select the GCP `default` cluster using the `cluster_id` returned by `list_clusters`. Worker allocation is configured in Studio, keeping the application script short. The current MCP tool has no Resume action; use Studio's Resume control.

## Verification status

The restored September 20 demo passed local failure/recovery tests using real video decoding and a deterministic test encoder. The inference sequence was `[1, 2, 3]`, then `[4, 5, 6, 7]`, then `[]` on a completed rerun. These tests did not load Qwen weights.

On September 24, 2026, the earlier seven-window Studio run on `aws-cpu` loaded the real Qwen weights with CPU PyTorch and reached the intended failure. Studio reported `rows_total: 7`, `rows_processed: 3`, and `rows_generated: 3`, followed by `RuntimeError: Demo interruption before embedding window 4`. That run used commit `010e286afb5d9beee309121a1d5ff42c5b2330dc`.

The full-input single-machine run was canceled at the user's request after Studio reported 57 processed rows and 56 generated rows, to switch to 300 distributed workers. The distributed failure, checkpoint reuse on Resume, and final 12,219-row dataset remain to be checked in Studio.
