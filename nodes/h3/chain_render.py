# APNext H3 Music Video Chain Render - carry the last frames of each scene into the next
#
# ComfyUI's list processing renders the writer's scenes as independent clips:
# scene N+1 cannot see scene N's frames, because both are items of the same
# list. Real continuity needs the previous *render*, so this node does what
# the Contex Loop pack does with its loop - but inside one execution, with the
# stock nodes: for every scene it builds the Ref2VA conditioning, injects the
# song piece into the audio latent (masked), pins the last `context_frames`
# of the previous delivered clip to the head of the new latent when the cut
# plan says the take CONTINUES, samples, decodes, trims the pinned head (and
# the grid padding) off, saves the clip, and keeps its tail for the next one.
#
# Which boundaries continue is the Cut Plan's call: a cut placed on a drop,
# a section start, a stop or one of your taps stays a hard cut (that is where
# the sync lives); a cut placed on a mere onset / downbeat / lyric line turns
# into a continuing take. Or force it either way with `continuity`.
#
# Mechanisms, in order of preference:
#   1. MiniMaxH3SongMaskedAVContext (ComfyUI-H3-Motion-Context-MultiRef): masked
#      AV prefix - previous frames encoded into the head of the target latent,
#      protected by a nested denoise mask, with the song audio masked in at the
#      matching offset. Same node the masked-audio workflow already uses.
#   2. MiniMaxH3AddGuide (core): the previous tail anchored as a guide clip at
#      frame 0. Works without any third-party pack; audio is then generated.
#
# The first scene, and every scene after a hard cut, renders exactly as the
# independent pipeline would. Continuing scenes render on a longer H3 grid
# length (the pinned prefix plus the piece), and the delivered clip is cut
# back to the piece's own frames, so the audio piece and the video still match
# to the frame.

import math
import re

from ...utils.constants import CUSTOM_CATEGORY
from .clip_save import save_frames
from .music_support import FPS, FRAME_STEP, MAX_FRAMES

CONTINUITY_MODES = [
    "cut plan decides (continue over soft cuts, hard cut on drops / sections / taps)",
    "flow everywhere (one continuous take)",
    "cut everywhere (independent clips)",
]

# cut reasons (see music_support.cut_reason) that are *soft*: the take may continue
FLOW_REASONS = ("onset", "downbeat", "lyric line")
MAX_REFS = 4


def snap_run(n):
    """Largest valid H3 clip run (5 + 17k) not above n, at least 5."""
    n = int(n)
    if n < 5:
        return 5
    return ((n - 5) // FRAME_STEP) * FRAME_STEP + 5


def grid_up(n):
    """Smallest valid H3 frame count (5 + 17k) at or above n."""
    n = int(n)
    return n if (n - 5) % FRAME_STEP == 0 and n >= 5 else 5 + FRAME_STEP * math.ceil(max(0, n - 5) / FRAME_STEP)


def flow_flags_from_cut_plan(text, n):
    """
    [continues?] per scene index (0-based). Scene i continues when the cut that
    ENDS scene i (line i in the plan, 1-based) sits on a soft reason.
    """
    reasons = {}
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(\d+)\s+\d+:\d.*?\bcut:\s*(.+?)\s*$", line)
        if m:
            reasons[int(m.group(1))] = m.group(2).lower()
    flags = [False] * n
    for i in range(1, n):
        reason = reasons.get(i)
        flags[i] = bool(reason) and any(reason.startswith(k) for k in FLOW_REASONS)
    return flags


def decide_flow(continuity, cut_plan, n):
    if n <= 0:
        return []
    mode = str(continuity or CONTINUITY_MODES[0])
    if mode.startswith("flow"):
        return [False] + [True] * (n - 1)
    if mode.startswith("cut everywhere"):
        return [False] * n
    if (cut_plan or "").strip():
        return flow_flags_from_cut_plan(cut_plan, n)
    return [False] * n


def _first(value, default=None):
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _node(name):
    import nodes as comfy_nodes
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"H3 Chain Render: node '{name}' is not available in this ComfyUI.")
    return cls


def _call(cls, **kwargs):
    """Run a node class's FUNCTION (v1 instance method or v3 classmethod) and return its outputs tuple."""
    fn_name = getattr(cls, "FUNCTION", "execute")
    fn = getattr(cls, fn_name, None)
    if fn is None:
        raise RuntimeError(f"H3 Chain Render: {cls.__name__} has no '{fn_name}'.")
    try:
        out = fn(**kwargs)                       # v3: classmethod
    except TypeError as exc:
        if "self" not in str(exc) and "positional" not in str(exc):
            raise
        out = getattr(cls(), fn_name)(**kwargs)  # v1: instance method
    if hasattr(out, "args"):
        return tuple(out.args)
    if isinstance(out, dict) and "result" in out:
        return tuple(out["result"])
    return tuple(out) if isinstance(out, (list, tuple)) else (out,)


class H3MusicVideoChainRender:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "master_audio": ("AUDIO", {
                "tooltip": (
                    "The whole song (Load Audio). With it, every scene's audio is the song piece masked "
                    "into the latent (the masked-audio workflow), and a continuing scene's pinned head "
                    "carries the previous piece's tail audio too. Without it, H3 generates audio."
                ),
            }),
            "cut_plan": ("STRING", {
                "forceInput": True,
                "tooltip": (
                    "The Cut Plan text (also wired into the writer). Its `cut: ...` reasons decide which "
                    "boundaries continue: onset / downbeat / lyric line continue the take, drops, "
                    "section starts, stops and your taps stay hard cuts."
                ),
            }),
        }
        for k in range(1, MAX_REFS + 1):
            optional[f"ref_image_{k}"] = ("IMAGE", {"tooltip": f"Reference picture <Picture {k}> for every scene (Ref2VA)."})
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The H3 model, with whatever attention / SoL patches you use."}),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "H3 audio VAE."}),
                "scenes": ("STRING", {"forceInput": True, "tooltip": "The writer's `scenes` list."}),
                "lengths": ("INT", {"forceInput": True, "tooltip": "The writer's `lengths` list (frames per scene)."}),
                "audio_segments": ("AUDIO", {"forceInput": True, "tooltip": "The writer's `audio_segments` list (one song piece per scene)."}),
                "clip_starts": ("FLOAT", {"forceInput": True, "tooltip": "The writer's `clip_starts` list (seconds into the song)."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Scene k samples with seed + k."}),
                "context_frames": ("INT", {
                    "default": 22, "min": 5, "max": 141, "step": 1,
                    "tooltip": (
                        "How many of the previous clip's last frames are pinned to the head of a continuing "
                        "scene. Snapped to H3's valid runs (5, 22, 39, 56, ...). 22 = ~0.9 s of motion; 39 "
                        "also aligns the audio clock."
                    ),
                }),
                "continuity": (CONTINUITY_MODES, {"default": CONTINUITY_MODES[0]}),
                "filename_prefix": ("STRING", {"default": "video/MiniMax_H3"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.01}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_paths", "report")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Render the writer's scenes one after another, carrying the last frames of each clip into "
        "the next where the cut plan lets the take continue. Ref2VA conditioning, masked song audio, "
        "pinned head, sample, decode, trim, save - per scene, inside one node."
    )

    def render(self, model, clip, vae, audio_vae, scenes, lengths, audio_segments, clip_starts,
               width, height, sampler, sigmas, seed, context_frames, continuity, filename_prefix, fps,
               master_audio=None, cut_plan=None, **refs):
        import torch
        import comfy.model_management as mm

        model, clip, vae, audio_vae = _first(model), _first(clip), _first(vae), _first(audio_vae)
        sampler, sigmas = _first(sampler), _first(sigmas)
        width, height = int(_first(width, 1344)), int(_first(height, 768))
        seed = int(_first(seed, 0))
        fps_v = float(_first(fps, FPS))
        prefix = str(_first(filename_prefix, "video/MiniMax_H3"))
        continuity = str(_first(continuity, CONTINUITY_MODES[0]))
        ctx = snap_run(int(_first(context_frames, 22)))
        master = _first(master_audio)
        plan_text = str(_first(cut_plan, "") or "")
        scenes, lengths = _as_list(scenes), [int(x) for x in _as_list(lengths)]
        pieces, starts = _as_list(audio_segments), [float(x) for x in _as_list(clip_starts)]
        n = min(len(scenes), len(lengths), len(pieces), len(starts))
        if n == 0:
            raise ValueError("H3 Chain Render: no scenes - wire the writer's scenes / lengths / audio_segments / clip_starts.")

        ref_images = {}
        for k in range(1, MAX_REFS + 1):
            img = _first(refs.get(f"ref_image_{k}"))
            if img is not None:
                ref_images[f"ref_image_{len(ref_images)}"] = img

        flags = decide_flow(continuity, plan_text, n)
        ref2va = _node("MiniMaxH3ReferenceToVideo")
        guider_cls, noise_cls = _node("BasicGuider"), _node("RandomNoise")
        sampler_cls, decode_cls = _node("SamplerCustomAdvanced"), _node("VAEDecode")
        import nodes as comfy_nodes
        masked_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3SongMaskedAVContext")
        guide_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3AddGuide")
        if master is not None and masked_cls is None:
            print("⚠️ H3 Chain Render: master_audio is connected but ComfyUI-H3-Motion-Context-MultiRef is not "
                  "installed - audio will be generated and continuity uses the core guide.")
        mechanism = "masked AV prefix" if (master is not None and masked_cls) else ("core guide clip" if guide_cls else "none")

        paths, lines = [], []
        lines.append(f"CHAIN RENDER | {n} scene(s) | context {ctx} frames | continuity: {continuity.split(' (')[0]} | mechanism: {mechanism}")
        prev_tail = None
        for i in range(n):
            length = lengths[i]
            flow = bool(flags[i]) and prev_tail is not None and mechanism != "none"
            if flow and grid_up(length + ctx) > MAX_FRAMES:
                print(f"⚠️ H3 Chain Render: scene {i + 1:02d} is {length} frames - with {ctx} pinned frames the render "
                      f"would exceed H3's {MAX_FRAMES}-frame range, so it opens on a hard cut instead. Shorter "
                      "pieces (or fewer context frames) let it continue.")
                flow = False
            render_len = grid_up(length + ctx) if flow else grid_up(length)
            head = ctx if flow else 0
            tail_pad = render_len - head - length
            start = starts[i] - (head / fps_v)
            print(f"🎞️ H3 Chain Render | scene {i + 1:02d}/{n} | {'CONTINUES' if flow else 'hard cut'} | "
                  f"{length} frames{f' (+{head} pinned, +{tail_pad} pad)' if flow else ''} | song @ {start:.2f}s")

            cond, latent = _call(ref2va, clip=clip, vae=vae, audio_vae=audio_vae, prompt=scenes[i],
                                 width=width, height=height, length=render_len, ref_image_size="match",
                                 ref_images=ref_images or None)[:2]
            trim = 0
            if master is not None and masked_cls is not None:
                if flow:
                    latent, trim, _clip_audio = _call(
                        masked_cls, latent=latent, audio_vae=audio_vae, master_audio=master,
                        clip_start_seconds=start, context_length=ctx, source_fps=fps_v, crop="disabled",
                        vae=vae, source_frames=prev_tail,
                    )[:3]
                    trim = int(trim)
                else:
                    latent = _call(masked_cls, latent=latent, audio_vae=audio_vae, master_audio=master,
                                   clip_start_seconds=start, context_length=0, source_fps=fps_v, crop="disabled")[0]
            elif flow and guide_cls is not None:
                cond = _call(guide_cls, positive=cond, latent=latent, frame_idx=0, vae=vae, image=prev_tail)[0]
                trim = ctx

            guider = _call(guider_cls, model=model, conditioning=cond)[0]
            noise = _call(noise_cls, noise_seed=seed + i)[0]
            sampled = _call(sampler_cls, noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)[0]
            images = _call(decode_cls, vae=vae, samples=sampled)[0]
            if images.dim() == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            total = int(images.shape[0])
            head_cut = trim if flow else 0
            delivered = images[head_cut:head_cut + length]
            if delivered.shape[0] < length:
                print(f"⚠️ H3 Chain Render: scene {i + 1:02d} decoded {total} frames, delivering {delivered.shape[0]} of {length}.")
            path = save_frames(delivered, pieces[i], f"{prefix}_s{i + 1:02d}", fps_v, "mp4")
            paths.append(path)
            lines.append(f"{i + 1:02d}  {'continues' if flow else 'cut      '}  {delivered.shape[0]} frames  "
                         f"{'head ' + str(head_cut) + ' trimmed' if flow else ''}  -> {path}")
            keep = min(ctx, int(delivered.shape[0]))
            prev_tail = delivered[-keep:].detach().clone()
            del images, delivered, sampled, latent, cond, guider, noise
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            mm.soft_empty_cache()

        report = "\n".join(lines)
        print(report)
        return {"ui": {"text": [report]}, "result": (paths, report)}
