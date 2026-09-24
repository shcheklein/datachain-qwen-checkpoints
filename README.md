# DataChain Qwen checkpoint demo

Compute one Qwen video embedding per window, interrupt the job before window 4, then resume using DataChain checkpoints.

## Files

- `qwen_checkpoint_demo.py`: the 30-line job shown in the demo.
- `qwen_encoder.py`: native DataChain video decoding, DataModel schemas, and the pinned Qwen3-VL-Embedding-8B encoder.
- `requirements-cpu.txt`: CPU dependencies for Python 3.11.
- `LICENSE-QWEN.txt`: license for the upstream Qwen implementation.

## Run in Studio

1. Set the repository to `https://github.com/shcheklein/datachain-qwen-checkpoints@main`.
2. Paste `qwen_checkpoint_demo.py` into the job editor.
3. Use the CPU cluster, Python 3.11, zero distributed workers, and the contents of `requirements-cpu.txt`.
4. Run. The job is designed to compute three embeddings and raise `Demo interruption before embedding window 4`.
5. Remove the two-line `if sampling.index == 3` failure block from the job editor, then click **Resume** on that failed job.

Keep the source version, filter, output name, and return schema unchanged. The resumed job should compute windows 4–7 and save seven rows. Model loading can happen again; checkpoints retain computed results, not an in-memory model. Buffered work that was not committed may repeat after a hard interruption.

The input is `@shcheklein.default.l2_egodex_qwen_7165f0d173@1.0.0`, filtered to `basic_pick_place/244`. Each window uses 16 sampled frames and produces one normalized 4,096-dimensional embedding. The output is `@shcheklein.default.qwen_checkpoint_demo`.

## Studio MCP

The connected `run_job` tool accepts a `repository` URL with an optional `@branch` or `@tag`. It also requires `query`, containing the Python source from `qwen_checkpoint_demo.py`. Its `query_name` is a display name, not a repository file selector. The encoder is imported from the cloned repository.

Pass the requirements file contents through `requirements`, set `python_version` to `3.11`, and use `workers: 0`. Select a cluster using the `cluster_id` returned by `list_clusters`. The current MCP tool has no Resume action; use Studio's Resume control.

## Verification status

The restored September 20 demo passed local failure/recovery tests using real video decoding and a deterministic test encoder. The inference sequence was `[1, 2, 3]`, then `[4, 5, 6, 7]`, then `[]` on a completed rerun. These tests did not load Qwen weights.

On September 24, 2026, a repository-backed Studio run on `aws-cpu` loaded the real Qwen weights with CPU PyTorch and reached the intended failure. Studio reported `rows_total: 7`, `rows_processed: 3`, and `rows_generated: 3`, followed by `RuntimeError: Demo interruption before embedding window 4`. The run used commit `010e286afb5d9beee309121a1d5ff42c5b2330dc`.

Recovery with Studio's **Resume** control is still pending. Reuse of the three completed embeddings and the final seven-row dataset have not yet been verified in Studio.
