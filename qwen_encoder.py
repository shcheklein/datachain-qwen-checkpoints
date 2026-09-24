"""Pinned Qwen video encoder used by the checkpoint demo.

Uses native DataChain VideoFragment/VideoFrame objects and DataModel schemas.
Inference and frame preprocessing are retained from the completed EgoDex job.
Model architecture and pooling follow QwenLM/Qwen3-VL-Embedding (Apache-2.0).
This module contains no checkpoint, retry, or completed-row filtering logic.
"""

import hashlib
import json
import math

import datachain as dc
import numpy as np
from PIL import Image



class Recipe(dc.DataModel):
    model_config = {"frozen": True}
    implementation: str = "egodex-qwen-video-v2-fragments"
    model_id: str = "Qwen/Qwen3-VL-Embedding-8B"
    revision: str = "2c4565515e0f265c6511776e7193b22c0968ddc7"
    dimensions: int = 4096
    window_seconds: float = 4.0
    stride_seconds: float = 2.0
    frames_per_window: int = 16
    max_frame_pixels: int = 262144
    max_tokens: int = 8192
    instruction: str = "Represent the user's input."
    resize: str = "PIL-bicubic-aspect-preserving-multiple-of-32"
    pooling: str = "last-attended-token-l2-normalized-float32"


RECIPE = Recipe()


class Sampling(dc.DataModel):
    index: int
    start_frame: int
    end_frame: int  # exclusive
    source_fps: float
    frame_indices: list[int]


class Runtime(dc.DataModel):
    device: str
    dtype: str
    torch_version: str
    cpu_threads: int


class Embedding(dc.DataModel):
    vector: list[float]
    sampled_timestamps_s: list[float]
    recipe_id: str
    recipe: Recipe
    runtime: Runtime | None = None


def recipe_id() -> str:
    encoded = json.dumps(RECIPE.model_dump(), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:10]


def decode_window(fragment: dc.VideoFragment, sampling: Sampling):
    """Read native VideoFrame objects from the fragment's original VideoFile."""
    if not math.isclose(
        fragment.start, sampling.start_frame / sampling.source_fps, abs_tol=1e-9
    ):
        raise ValueError("Fragment start does not match frame sampling")
    if not math.isclose(
        fragment.end, sampling.end_frame / sampling.source_fps, abs_tol=1e-9
    ):
        raise ValueError("Fragment end does not match frame sampling")
    file = fragment.video
    wanted = set(sampling.frame_indices)
    frames = {}
    timestamps = {}
    frame: dc.VideoFrame
    for frame in file.get_frames(start=sampling.start_frame, end=sampling.end_frame):
        if frame.frame in wanted:
            frames[frame.frame] = Image.fromarray(frame.get_np()).convert("RGB")
            timestamps[frame.frame] = frame.timestamp
    missing = wanted - frames.keys()
    if missing:
        raise ValueError(
            f"{file.path}: missing frames {sorted(missing)}; "
            "check MP4/HDF5 frame-count agreement"
        )
    return (
        [frames[i] for i in sampling.frame_indices],
        [timestamps[i] for i in sampling.frame_indices],
    )


def embedding_model_class():
    # Match the released checkpoint's model.* keys. Loading the bare Qwen3VLModel
    # directly does not use this key layout in Transformers 4.57.3.
    # Architecture/last-token pooling follow QwenLM/Qwen3-VL-Embedding (Apache-2.0).
    from transformers import Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def forward(self, **kwargs):
            return self.model(**kwargs)

    return Qwen3VLForEmbedding


class QwenEncoder:
    def __init__(self, model, processor, runtime=None):
        self.model = model
        self.processor = processor
        self.runtime = runtime

    def prepare(self, *, frames=None, fragment=None, sampling=None, text=None):
        import torch
        from qwen_vl_utils import smart_resize

        if (frames is None) == (text is None):
            raise ValueError("Supply either video frames or query text")
        processor_args = {}
        if frames is not None:
            if (
                fragment is None
                or sampling is None
                or len(frames) != len(sampling.frame_indices)
            ):
                raise ValueError("Video frames and window metadata must agree")
            height, width = smart_resize(
                frames[0].height,
                frames[0].width,
                factor=32,
                min_pixels=4096,
                max_pixels=RECIPE.max_frame_pixels,
            )
            arrays = [
                np.asarray(im.resize((width, height), Image.Resampling.BICUBIC))
                for im in frames
            ]
            # One video containing ordered frames, not 16 unrelated image inputs.
            video = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
            content = [{"type": "video", "video": frames}]
            processor_args = {
                "videos": [video],
                "video_metadata": [
                    {
                        "fps": sampling.source_fps,
                        "frames_indices": [
                            i - sampling.start_frame for i in sampling.frame_indices
                        ],
                        "total_num_frames": sampling.end_frame - sampling.start_frame,
                        "video_backend": "datachain",
                    }
                ],
                "do_sample_frames": False,
                "do_resize": False,
            }
        else:
            content = [{"type": "text", "text": text}]
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": RECIPE.instruction}],
            },
            {"role": "user", "content": content},
        ]
        formatted = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        # Never truncate video tokens or substitute a text-only embedding on errors.
        inputs = self.processor(
            text=[formatted],
            padding=True,
            truncation=False,
            return_tensors="pt",
            **processor_args,
        )
        if inputs["input_ids"].shape[1] > RECIPE.max_tokens:
            raise ValueError("Input exceeds the configured token budget")
        return inputs

    def encode(self, **kwargs) -> list[float]:
        import torch
        import torch.nn.functional as F

        inputs = self.prepare(**kwargs).to(self.model.device)
        with torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            mask = inputs["attention_mask"]
            last = mask.shape[1] - 1 - mask.flip(1).argmax(1)
            vector = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
            vector = vector.float()
            if not torch.isfinite(vector).all() or (vector.norm(dim=-1) == 0).any():
                raise ValueError("Qwen produced a non-finite or zero embedding")
            vector = F.normalize(vector, p=2, dim=-1)[0].cpu().tolist()
        if len(vector) != RECIPE.dimensions:
            raise ValueError(f"Unexpected embedding dimension: {len(vector)}")
        return vector

    def embed_video(self, fragment: dc.VideoFragment, sampling: Sampling) -> Embedding:
        frames, timestamps = decode_window(fragment, sampling)
        vector = self.encode(frames=frames, fragment=fragment, sampling=sampling)
        return Embedding(
            vector=vector,
            sampled_timestamps_s=timestamps,
            recipe_id=recipe_id(),
            recipe=RECIPE,
            runtime=self.runtime,
        )



def load_encoder() -> QwenEncoder:
    """Load the pinned 8B encoder on CPU, once for this sequential worker."""
    import torch
    from transformers import Qwen3VLProcessor

    torch.set_num_threads(16)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    runtime = Runtime(
        device="cpu",
        dtype="bfloat16",
        torch_version=str(torch.__version__),
        cpu_threads=16,
    )
    model, info = embedding_model_class().from_pretrained(
        RECIPE.model_id,
        revision=RECIPE.revision,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        output_loading_info=True,
    )
    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise RuntimeError(f"Checkpoint architecture mismatch: {info}")
    processor = Qwen3VLProcessor.from_pretrained(
        RECIPE.model_id,
        revision=RECIPE.revision,
        padding_side="right",
        trust_remote_code=False,
    )
    return QwenEncoder(model.eval(), processor, runtime)
