from collections.abc import Sequence
import dataclasses
import hashlib
import pathlib
import shutil
import subprocess
from typing import Any

import av
import cv2
import imageio_ffmpeg
import numpy as np
import safetensors

from openpi.training import token_budget


@dataclasses.dataclass(frozen=True)
class VisualTokenLayout:
    """Index mapping helper between camera patch grids and routed/full visual token sequences."""

    num_cameras: int = 3
    grid_height: int = 16
    grid_width: int = 16
    image_height: int = 224
    image_width: int = 224

    def __post_init__(self) -> None:
        if self.num_cameras <= 0 or self.grid_height <= 0 or self.grid_width <= 0:
            raise ValueError("num_cameras, grid_height, and grid_width must be positive")
        if self.image_height % self.grid_height != 0 or self.image_width % self.grid_width != 0:
            raise ValueError("image dimensions must be divisible by grid dimensions")

    @property
    def patches_per_camera(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def total_visual_tokens(self) -> int:
        return self.num_cameras * self.patches_per_camera

    @property
    def patch_height(self) -> int:
        return self.image_height // self.grid_height

    @property
    def patch_width(self) -> int:
        return self.image_width // self.grid_width

    def patch_index_to_camera_rc(self, index: int) -> tuple[int, int, int]:
        idx = int(index)
        if idx < 0 or idx >= self.total_visual_tokens:
            raise IndexError(f"Patch index {idx} out of range [0, {self.total_visual_tokens})")
        cam_idx, rem = divmod(idx, self.patches_per_camera)
        row, col = divmod(rem, self.grid_width)
        return cam_idx, row, col

    def patch_bbox(self, index: int) -> tuple[int, int, int, int, int]:
        cam_idx, row, col = self.patch_index_to_camera_rc(index)
        y0 = row * self.patch_height
        x0 = col * self.patch_width
        return cam_idx, y0, x0, y0 + self.patch_height, x0 + self.patch_width

    def scatter_routed_attention(
        self,
        routed_attention: np.ndarray,
        selected_indices: np.ndarray | None = None,
        *,
        hard_tokens: int = 0,
        soft_tokens: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scatter Hard token attention back to original patch positions and sum Soft token attention mass."""
        attn = np.asarray(routed_attention, dtype=np.float32)
        squeeze = attn.ndim == 1
        if squeeze:
            attn = attn[None, :]
        if attn.ndim != 2:
            raise ValueError(f"routed_attention must be 1D or 2D, got shape {attn.shape}")

        bsize, seq_len = attn.shape
        if seq_len != hard_tokens + soft_tokens:
            raise ValueError(
                f"routed_attention length ({seq_len}) does not equal hard_tokens ({hard_tokens}) + soft_tokens ({soft_tokens})"
            )

        hard_map = np.zeros((bsize, self.total_visual_tokens), dtype=np.float32)
        if hard_tokens > 0:
            if selected_indices is None:
                raise ValueError("selected_indices is required when hard_tokens > 0")
            sel = np.asarray(selected_indices, dtype=np.int64)
            if sel.ndim == 1:
                sel = sel[None, :]
            if sel.shape != (bsize, hard_tokens):
                raise ValueError(f"selected_indices shape {sel.shape} != ({bsize}, {hard_tokens})")
            if np.any(sel < 0) or np.any(sel >= self.total_visual_tokens):
                raise ValueError("selected_indices out of bounds")
            np.put_along_axis(hard_map, sel, attn[:, :hard_tokens], axis=1)

        soft_mass = attn[:, hard_tokens:].sum(axis=1) if soft_tokens > 0 else np.zeros(bsize, dtype=np.float32)
        if squeeze:
            return hard_map[0], soft_mass[0]
        return hard_map, soft_mass

    def soft_source_contribution(self, assignment: np.ndarray) -> np.ndarray:
        """Compute mean_g assignment[..., i, g] over soft output slots for each original patch i."""
        assign = np.asarray(assignment, dtype=np.float32)
        if assign.shape[-2] != self.total_visual_tokens:
            raise ValueError(
                f"assignment patch dimension {assign.shape[-2]} != total_visual_tokens {self.total_visual_tokens}"
            )
        if assign.shape[-1] == 0:
            return np.zeros(assign.shape[:-1], dtype=np.float32)
        return assign.mean(axis=-1)


@dataclasses.dataclass(frozen=True)
class MinimalCoverageSelection:
    normalized_probs: np.ndarray
    selected_mask: np.ndarray
    selected_indices: np.ndarray
    num_selected: int
    achieved_coverage: float


def select_minimal_coverage_set(
    relevance_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    coverage: float,
) -> MinimalCoverageSelection:
    """Select the minimal set of valid visual tokens reaching cumulative coverage on a single frame."""
    if not (0.0 < float(coverage) <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    probs_2d, mask_2d, _ = token_budget.validate_And_normalize_relevance(
        np.asarray(relevance_1d, dtype=np.float64)[None, :],
        np.asarray(valid_mask_1d, dtype=bool)[None, :],
    )
    probs = probs_2d[0]
    mask = mask_2d[0]
    total_tokens = probs.shape[0]
    valid_count = int(mask.sum())
    indices = np.arange(total_tokens, dtype=np.int64)

    order = np.lexsort((indices, -probs, ~mask))[:valid_count]
    sorted_probs = probs[order]
    csum = np.cumsum(sorted_probs, dtype=np.float64)
    csum = csum / csum[-1]
    k = min(int(np.searchsorted(csum >= (float(coverage) - 1e-12), True, side="left")) + 1, valid_count)

    chosen = np.sort(order[:k])
    selected_mask = np.zeros(total_tokens, dtype=bool)
    selected_mask[chosen] = True

    verify_minimal_coverage_selection(probs, mask, selected_mask, coverage)
    return MinimalCoverageSelection(
        normalized_probs=probs.astype(np.float32),
        selected_mask=selected_mask,
        selected_indices=chosen,
        num_selected=int(k),
        achieved_coverage=float(csum[k - 1]),
    )


def verify_minimal_coverage_selection(
    normalized_probs_1d: np.ndarray,
    valid_mask_1d: np.ndarray,
    selected_mask_1d: np.ndarray,
    coverage: float,
) -> None:
    """Verify mask exclusion and minimal coverage property on a single frame."""
    probs = np.asarray(normalized_probs_1d, dtype=np.float64)
    valid = np.asarray(valid_mask_1d, dtype=bool)
    selected = np.asarray(selected_mask_1d, dtype=bool)

    if np.any(selected & ~valid):
        raise ValueError("Selected coverage mask contains invalid/masked camera tokens")
    k = int(selected.sum())
    if k <= 0:
        raise ValueError("Selected coverage mask is empty")

    selected_mass = float(probs[selected].sum())
    if selected_mass + 1e-6 < float(coverage):
        raise ValueError(f"Selected coverage mass ({selected_mass:.6f}) is below target ({coverage})")

    if k > 1:
        min_selected_prob = float(probs[selected].min())
        if (selected_mass - min_selected_prob) >= float(coverage) + 1e-6:
            raise ValueError("Selected token set is not minimal for the requested coverage")


def compute_action_out_proj_fingerprint(weight_path: str | pathlib.Path) -> str:
    """Compute a deterministic SHA256 fingerprint of action_out_proj parameters in a checkpoint."""
    path = pathlib.Path(weight_path)
    safetensors_path = path / "model.safetensors" if path.is_dir() else path
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Checkpoint safetensors not found at {safetensors_path}")

    hasher = hashlib.sha256()
    with safetensors.safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        keys = sorted(k for k in f.keys() if k.startswith("action_out_proj."))
        if not keys:
            raise ValueError(f"No action_out_proj.* parameters found in {safetensors_path}")
        for key in keys:
            tensor = f.get_tensor(key).detach().cpu().float().contiguous()
            hasher.update(key.encode("utf-8"))
            hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a preprocessed [-1, 1] or [0, 1] or uint8 image to [H, W, 3] uint8 RGB."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype == np.uint8:
        return arr.copy()
    arr_f = arr.astype(np.float32)
    if arr_f.min() < -0.01 or arr_f.max() <= 1.05:
        if arr_f.min() < -0.01:
            arr_f = (arr_f + 1.0) * 0.5
        arr_f = np.clip(arr_f, 0.0, 1.0) * 255.0
    return np.clip(np.round(arr_f), 0, 255).astype(np.uint8)


def _render_single_camera_overlay(
    rgb_uint8: np.ndarray,
    patch_probs: np.ndarray,
    patch_selected: np.ndarray,
    *,
    camera_valid: bool,
    color_max: float,
    layout: VisualTokenLayout,
) -> np.ndarray:
    base = image_to_uint8(rgb_uint8)
    if base.shape[:2] != (layout.image_height, layout.image_width):
        base = cv2.resize(base, (layout.image_width, layout.image_height), interpolation=cv2.INTER_LINEAR)

    if not camera_valid:
        dimmed = (base.astype(np.float32) * 0.25).astype(np.uint8)
        cv2.putText(dimmed, "MASKED", (56, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        return dimmed

    grid_probs = np.asarray(patch_probs, dtype=np.float32).reshape(layout.grid_height, layout.grid_width)
    norm_grid = np.clip(grid_probs / max(float(color_max), 1e-12), 0.0, 1.0)
    # Nearest-neighbor upsampling from 16x16 to 224x224.
    upsampled = np.repeat(np.repeat(norm_grid, layout.patch_height, axis=0), layout.patch_width, axis=1)
    heat_u8 = np.clip(np.round(upsampled * 255.0), 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    blended = np.clip(
        np.round(base.astype(np.float32) * 0.45 + heat_rgb.astype(np.float32) * 0.55),
        0,
        255,
    ).astype(np.uint8)

    sel_grid = np.asarray(patch_selected, dtype=bool).reshape(layout.grid_height, layout.grid_width)
    ph, pw = layout.patch_height, layout.patch_width
    for r in range(layout.grid_height):
        for c in range(layout.grid_width):
            if sel_grid[r, c]:
                y0, x0 = r * ph, c * pw
                y1, x1 = y0 + ph - 1, x0 + pw - 1
                cv2.rectangle(blended, (x0, y0), (x1, y1), (255, 230, 0), 1)
    return blended


def render_attention_frame(
    camera_images: Sequence[np.ndarray],
    camera_masks: Sequence[bool],
    selections_by_source: dict[str, MinimalCoverageSelection],
    *,
    episode_id: int,
    frame_idx: int,
    coverage: float,
    color_max_multiplier: float = 8.0,
    camera_names: Sequence[str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    layout: VisualTokenLayout | None = None,
) -> np.ndarray:
    """Render a multi-row diagnostic frame for one step across cameras and teacher attention sources."""
    tok_layout = VisualTokenLayout(num_cameras=len(camera_images)) if layout is None else layout
    p_per_cam = tok_layout.patches_per_camera
    valid_patches = sum(p_per_cam for is_valid in camera_masks if bool(is_valid))
    color_max = float(color_max_multiplier) / float(max(valid_patches, 1))

    sources = [s for s in ("vit", "vlm", "action") if s in selections_by_source]
    header_h = 44
    row_label_w = 136
    cell_h = tok_layout.image_height
    cell_w = tok_layout.image_width
    num_rows = 1 + len(sources)
    total_h = header_h + num_rows * cell_h
    total_w = row_label_w + tok_layout.num_cameras * cell_w
    if total_h % 2 != 0:
        total_h += 1
    if total_w % 2 != 0:
        total_w += 1

    canvas = np.full((total_h, total_w, 3), 20, dtype=np.uint8)
    counts_text = " | ".join(
        f"{src.upper()}: {selections_by_source[src].num_selected}" for src in sources
    )
    title = f"ep={episode_id} frame={frame_idx} cov={coverage:.2f} | {counts_text}"
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    # Row 0: raw preprocessed 224x224 camera inputs
    y_top = header_h
    cv2.putText(canvas, "Input RGB", (12, y_top + cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    for cam_idx, (img, cam_valid) in enumerate(zip(camera_images, camera_masks, strict=True)):
        x_left = row_label_w + cam_idx * cell_w
        tile = image_to_uint8(img)
        if not bool(cam_valid):
            tile = (tile.astype(np.float32) * 0.25).astype(np.uint8)
        cam_label = camera_names[cam_idx] if cam_idx < len(camera_names) else f"cam_{cam_idx}"
        cv2.putText(tile, cam_label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[y_top : y_top + cell_h, x_left : x_left + cell_w] = tile

    # Subsequent rows: ViT / VLM / Action heatmaps + yellow minimal-coverage patch boxes
    for row_idx, src in enumerate(sources, start=1):
        sel = selections_by_source[src]
        y_row = header_h + row_idx * cell_h
        label = f"{src.upper()} (K={sel.num_selected})"
        cv2.putText(
            canvas,
            label,
            (12, y_row + cell_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 230, 0),
            1,
            cv2.LINE_AA,
        )
        for cam_idx, (img, cam_valid) in enumerate(zip(camera_images, camera_masks, strict=True)):
            x_left = row_label_w + cam_idx * cell_w
            start = cam_idx * p_per_cam
            end = start + p_per_cam
            overlay = _render_single_camera_overlay(
                img,
                sel.normalized_probs[start:end],
                sel.selected_mask[start:end],
                camera_valid=bool(cam_valid),
                color_max=color_max,
                layout=tok_layout,
            )
            canvas[y_row : y_row + cell_h, x_left : x_left + cell_w] = overlay

    return canvas


def encode_h264_mp4(
    frames_rgb: Sequence[np.ndarray],
    output_path: str | pathlib.Path,
    *,
    fps: int = 10,
) -> dict[str, Any]:
    """Encode RGB frames to H.264 MP4 (avc1, yuv420p, +faststart) and verify full decode."""
    if not frames_rgb:
        raise ValueError("frames_rgb must be non-empty")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    target = pathlib.Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing video file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    first = np.asarray(frames_rgb[0], dtype=np.uint8)
    height, width, channels = first.shape
    if channels != 3 or height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Video frames must be even-sized RGB [H, W, 3], got {first.shape}")

    ffmpeg_bin = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        str(target),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames_rgb:
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.shape != (height, width, 3):
                raise ValueError(f"Inconsistent frame shape {arr.shape}, expected {(height, width, 3)}")
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        proc.stdin = None
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}: {stderr}")
    finally:
        if proc.stdin is not None:
            proc.stdin.close()

    return verify_encoded_video(target, expected_frames=len(frames_rgb), expected_fps=fps)


def verify_encoded_video(
    video_path: str | pathlib.Path,
    *,
    expected_frames: int,
    expected_fps: int = 10,
) -> dict[str, Any]:
    """Verify H.264/avc1/yuv420p encoding, moov faststart placement, and full frame decodability."""
    path = pathlib.Path(video_path)
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Encoded video is missing or empty: {path}")

    raw_bytes = path.read_bytes()
    moov_pos = raw_bytes.find(b"moov")
    mdat_pos = raw_bytes.find(b"mdat")
    if moov_pos < 0 or mdat_pos < 0 or moov_pos > mdat_pos:
        raise ValueError(f"MP4 file does not have faststart (moov before mdat): {path}")

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        codec_name = stream.codec_context.name
        pix_fmt = stream.codec_context.pix_fmt
        codec_tag = stream.codec_context.codec_tag
        width = stream.codec_context.width
        height = stream.codec_context.height
        decoded_count = 0
        for frame in container.decode(video=0):
            _ = frame.to_ndarray(format="rgb24")
            decoded_count += 1

    if codec_name != "h264":
        raise ValueError(f"Expected h264 codec, got '{codec_name}' in {path}")
    if pix_fmt != "yuv420p":
        raise ValueError(f"Expected yuv420p pixel format, got '{pix_fmt}' in {path}")
    if decoded_count != int(expected_frames):
        raise ValueError(f"Decoded frame count ({decoded_count}) != expected ({expected_frames}) in {path}")

    return {
        "codec": codec_name,
        "codec_tag": codec_tag,
        "pix_fmt": pix_fmt,
        "faststart": True,
        "fps": int(expected_fps),
        "width": int(width),
        "height": int(height),
        "num_frames": int(decoded_count),
    }
