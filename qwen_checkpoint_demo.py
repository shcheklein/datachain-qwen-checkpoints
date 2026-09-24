"""Embed all windows, fail after 1,000, remove the failure, then Resume."""

import datachain as dc

from qwen_encoder import Embedding, Sampling, load_encoder


def embed(
    row_id: int, fragment: dc.VideoFragment, sampling: Sampling, encoder
) -> Embedding:
    # Deliberate failure. Remove this block, then click Resume.
    if row_id == 1001:
        raise RuntimeError("Demo interruption after 1,000 embeddings")
    return encoder.embed_video(fragment, sampling)


(
    dc.read_dataset("@shcheklein.default.l2_egodex_qwen_7165f0d173", version="1.0.0")
    .mutate(row_id=dc.C("sys.id"))
    .order_by("row_id")
    .setup(encoder=load_encoder)
    .map(embedding=embed)
    .save(
        "@shcheklein.default.qwen_checkpoint_demo",
        attrs=["cast:sense", "scope:directory", "source:egodex"],
        description="Qwen video embeddings for all 12,219 EgoDex windows; CPU checkpoint demo.",
    )
)
