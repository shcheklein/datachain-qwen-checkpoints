"""Fail on the fourth video window, remove the failure, then Resume."""

import datachain as dc

from qwen_encoder import Embedding, Sampling, load_encoder

SOURCE = "@shcheklein.default.l2_egodex_qwen_7165f0d173"
EPISODE = "basic_pick_place/244"
OUTPUT = "@shcheklein.default.qwen_checkpoint_demo"


def embed(fragment: dc.VideoFragment, sampling: Sampling, encoder) -> Embedding:
    # Deliberate failure. Remove this block, then click Resume.
    if sampling.index == 3:
        raise RuntimeError("Demo interruption before embedding window 4")
    return encoder.embed_video(fragment, sampling)


(
    dc.read_dataset(SOURCE, version="1.0.0")
    .filter(dc.C("key") == EPISODE)
    .order_by("sampling.index")
    .setup(encoder=load_encoder)
    .map(embedding=embed)
    .save(
        OUTPUT,
        attrs=["cast:sense", "scope:sample", "source:egodex"],
        description="Qwen3-VL-Embedding-8B embeddings for seven EgoDex video windows. CPU/BF16 checkpoint demonstration.",
    )
)
