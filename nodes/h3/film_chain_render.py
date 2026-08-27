# APNext H3 Short Film Chain Render - carry picture AND sound between scenes
#
# The short-film pipeline renders the writer's scenes as independent clips,
# and H3 generates each clip's sound from nothing: the room tone, the score
# and whatever was mid-air at the previous cut all start over, so every
# scene change is a sound edit the film never asked for. The Music Video
# Chain Render solves its version of this with the master song; a film has
# no master track - its sound IS the previous clip's sound.
#
# So this node renders the scenes one after another inside one execution and,
# for every scene after the first, pins the last `context_frames` of the
# previous scene's SAMPLED LATENT - the video stream and the audio stream
# together - to the head of the new latent, protected by a noise mask, with
# the audio mask feathered over the last ticks so the join is a release, not
# a wall. The model reads the pinned run as "this clip's picture and sound so
# far" and continues both. No decode / re-encode round trip: the latent is
# copied straight across (ComfyUI-H3-Motion-Context-MultiRef's
# MiniMaxH3GeneratedAVMaskedContext - the node its own AV-extension
# workflows use). The delivered clip has the pinned head trimmed off, picture
# and sound by the same duration.
#
# Without that pack the core MiniMaxH3AddGuide still carries the previous
# picture (the last frames anchored as a guide clip); the sound is then
# generated per scene as before, and the report says so.
#
# Pair it with the Short Film Writer's `continuity_mode` = Continuous chain:
# the writer then opens every scene on the previous ending and its hand-off
# pass keeps the objects, positions and light the same across the join, so
# the words and the pixels agree.


import os

from ...utils.constants import CUSTOM_CATEGORY
from .chain_render import MAX_REFS, _as_list, _call, _first, _node, grid_up, snap_run
from .clip_save import save_frames
from .latent_store import project_key, save_scene_latent
from .music_support import FPS, MAX_FRAMES

FILM_CONTINUITY_MODES = [
    "flow everywhere (every scene continues the previous picture and sound)",
    "cut everywhere (independent clips - render as before)",
]

# H3 runs where the 24 fps picture and the 40 Hz audio latent share a
# boundary: 39, 90, 141, ... frames (39 + 51k). Only these pin the sound to
# the same instant as the picture.
AV_ALIGNED_RUNS = tuple(39 + 51 * k for k in range(0, 8))

# How the previous scene reaches the next one. The masked AV latent is the
# surest carry (both streams copied into the new latent under a noise mask),
# but the two grids only share a boundary every 51 frames, so it takes 39 /
# 90 / 141. The pack's Motion Context guide pins the previous frames as
# never-denoised conditioning rows - any H3 run (5, 22, 39, 56, 73, ...) -
# and pins the tail of the previous SOUND on the same timeline from the
# saved latent, so the sound still continues. The core guide clip carries
# the picture only.
CONTEXT_MODES = [
    "masked AV latent (picture + sound; 39 / 90 / 141 frames)",
    "motion context guide (picture + sound; any run: 5, 22, 39, 56, 73 ...)",
    "core guide clip (picture only; any run)",
]


def snap_context(mode, n):
    """The context run the chosen mechanism can actually pin."""
    return snap_av_run(n) if str(mode or "").startswith("masked") else max(5, snap_run(n))


def snap_av_run(n):
    """The largest AV-aligned context run (39, 90, 141, ...) not above n; 39 at least."""
    n = int(n)
    best = AV_ALIGNED_RUNS[0]
    for run in AV_ALIGNED_RUNS:
        if run <= n:
            best = run
    return best


def _fade(n, out=True):
    """Raised-cosine ramp of n samples: 1 -> 0 (out) or 0 -> 1 (in)."""
    import torch
    t = torch.linspace(0.0, 1.0, max(2, n))
    ramp = 0.5 - 0.5 * torch.cos(t * 3.141592653589793)
    return (1.0 - ramp) if out else ramp


def seam_join(track, piece, head_samples, xfade_samples, sr, declick_ms=5.0):
    """
    Append one scene's audio to the continuous track across the seam.

    `piece` is the scene's full decoded audio INCLUDING its pinned head (the
    previous scene's tail, `head_samples` long), cut to its delivered length.
    A continuing scene has, in that head, a second take of the sound the
    track already ends with - identical at first, bending into the new
    scene's continuation over the feathered end. So instead of a hard splice
    at the end of the head (a click, and a level or phase jump you hear as
    "a new clip"), the last `xfade_samples` of the track are blended into the
    same span of the piece with an equal-power crossfade, and the piece
    carries on from there. Total length is unchanged. A hard cut (no head)
    gets a 5 ms raised-cosine out/in so the splice itself cannot click.
    """
    import torch
    if track is None:
        return piece[..., head_samples:].clone()
    xf = int(min(xfade_samples, head_samples, track.shape[-1], piece.shape[-1]))
    if head_samples > 0 and xf > 8:
        a = track[..., -xf:]
        b = piece[..., head_samples - xf:head_samples].to(track.device, track.dtype)
        t = torch.linspace(0.0, 1.0, xf, device=track.device, dtype=track.dtype)
        wa = torch.cos(t * 3.141592653589793 / 2)      # equal power: wa^2 + wb^2 = 1
        wb = torch.sin(t * 3.141592653589793 / 2)
        blend = a * wa + b * wb
        return torch.cat([track[..., :-xf], blend, piece[..., head_samples:].to(track.device, track.dtype)], dim=-1)
    rest = piece[..., head_samples:].to(track.device, track.dtype).clone()
    n = max(2, int(round(declick_ms / 1000.0 * sr)))
    n = min(n, track.shape[-1], rest.shape[-1])
    if n >= 2:
        track = track.clone()
        track[..., -n:] = track[..., -n:] * _fade(n, out=True).to(track.device, track.dtype)
        rest[..., :n] = rest[..., :n] * _fade(n, out=False).to(track.device, track.dtype)
    return torch.cat([track, rest], dim=-1)


def save_track(track, sr, first_clip_path, prefix):
    """Write the continuous track as a wav next to the clips; returns the path or None."""
    try:
        import torchaudio
        folder = os.path.dirname(first_clip_path)
        name = os.path.basename(prefix.replace("\\", "/").rstrip("/")) or "H3"
        path = os.path.join(folder, f"{name}_audio.wav")
        torchaudio.save(path, track[0].detach().to("cpu").float(), int(sr))
        return path
    except Exception as exc:
        print(f"⚠️ H3: could not write the continuous audio track: {exc}")
        return None


def trim_audio(audio, trim_frames, frames_left, fps):
    """
    Drop the pinned head off a clip's decoded audio and cut its tail to
    exactly `frames_left / fps`. H3 rounds its audio grid UP (124 frames want
    206.67 audio steps, it allocates 207), so every clip ships ~8 ms more
    sound than picture - harmless alone, a growing click down a chain.
    """
    if audio is None:
        return None
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    cut = int(round(trim_frames / float(fps) * sr))
    if cut > 0:
        if cut >= int(waveform.shape[-1]):
            raise ValueError(f"H3 Film Chain Render: trimming {trim_frames} frames of audio leaves nothing.")
        waveform = waveform[..., cut:]
    want = int(round(frames_left / float(fps) * sr))
    if int(waveform.shape[-1]) > want:
        waveform = waveform[..., :want]
    return {"waveform": waveform, "sample_rate": sr}


class H3ShortFilmChainRender:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for k in range(1, MAX_REFS + 1):
            optional[f"ref_image_{k}"] = ("IMAGE", {"tooltip": f"Reference picture <Picture {k}> for every scene (Ref2VA)."})
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The H3 model, with whatever attention / SoL patches you use."}),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "H3 audio VAE - decodes each scene's sound."}),
                "scenes": ("STRING", {"forceInput": True, "tooltip": "The Short Film Writer's `scenes` list."}),
                "lengths": ("INT", {"forceInput": True, "tooltip": "The writer's `lengths` list (frames per scene)."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Scene k samples with seed + k."}),
                "context_frames": ("INT", {
                    "default": 39, "min": 5, "max": 141, "step": 1,
                    "tooltip": (
                        "How much of the previous scene is pinned to the head of the next one, in frames "
                        "(24 fps). Snapped to what context_mode can pin: the masked AV latent takes the "
                        "runs where picture and sound share a boundary - 39 (~1.6 s), 90 (~3.75 s), 141 "
                        "(~5.9 s); the guide modes take any H3 run - 5, 22, 39, 56, 73, 90, 107, 124, 141. "
                        "More context = a longer, surer continuation, and fewer frames left for the scene."
                    ),
                }),
                "audio_feather_ticks": ("INT", {
                    "default": 8, "min": 0, "max": 64,
                    "tooltip": (
                        "Half-cosine release over the last ticks of the pinned audio (40 ticks = 1 s). "
                        "0 = hard mask. 8 = 0.2 s: the model may bend the very end of the inherited sound "
                        "into what it generates next, which hides the seam."
                    ),
                }),
                "continuity": (FILM_CONTINUITY_MODES, {"default": FILM_CONTINUITY_MODES[0]}),
                "filename_prefix": ("STRING", {"default": "video/H3_film"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.01}),
                # appended last so saved workflows keep their widget positions
            },
            "optional": {
                **optional,
                # new since the first release: OPTIONAL with defaults, so a workflow saved (or a
                # browser tab loaded) before they existed still validates and runs
                "save_latents": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Write every scene's sampled latent to output/apnext_latents/<project>_sNN.pt so "
                        "H3 Scene Retake can render one scene again later, continuing from the previous "
                        "scene's real tail - picture and sound. ~10-30 MB per scene."
                    ),
                }),
                "seam_crossfade_ms": ("INT", {
                    "default": 400, "min": 0, "max": 1500, "step": 10,
                    "tooltip": (
                        "The `audio` output is ONE continuous track for the whole film. At every join where "
                        "the take continues, the previous scene's tail and the new scene's second take of it "
                        "(its pinned head) are blended with an equal-power crossfade this long - no splice, "
                        "no click, no level jump, and the total length is unchanged. Hard cuts get a 5 ms "
                        "declick. Wire `audio` into Scenes Join's `replace_audio` (or Save Audio); it is also "
                        "written as <prefix>_audio.wav next to the clips."
                    ),
                }),
                "context_mode": (CONTEXT_MODES, {
                    "default": CONTEXT_MODES[0],
                    "tooltip": (
                        "How the previous scene reaches the next. Masked AV latent: both streams copied "
                        "into the new latent under a noise mask - the surest carry, but only 39 / 90 / 141 "
                        "frames. Motion context guide: the previous frames pinned as never-denoised "
                        "conditioning (any H3 run) with the tail of the previous sound pinned on the same "
                        "timeline from the saved latent. Core guide clip: picture only, no pack needed."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("file_paths", "report", "audio")
    OUTPUT_IS_LIST = (True, False, False)
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Render a short film's scenes one after another, pinning the last frames AND the last sound of "
        "each scene into the head of the next (straight from the sampled latent), so the picture continues "
        "and the room tone / score / voice do not restart at every scene. Ref2VA, masked AV prefix, "
        "sample, decode, trim, save - per scene, inside one node."
    )

    def render(self, model, clip, vae, audio_vae, scenes, lengths, width, height, sampler, sigmas, seed,
               context_frames, audio_feather_ticks, continuity, filename_prefix, fps, save_latents=True,
               context_mode=None, seam_crossfade_ms=400, **refs):
        import torch
        import comfy.model_management as mm

        model, clip, vae, audio_vae = _first(model), _first(clip), _first(vae), _first(audio_vae)
        sampler, sigmas = _first(sampler), _first(sigmas)
        width, height = int(_first(width, 1344)), int(_first(height, 768))
        seed = int(_first(seed, 0))
        fps_v = float(_first(fps, FPS))
        prefix = str(_first(filename_prefix, "video/H3_film"))
        continuity = str(_first(continuity, FILM_CONTINUITY_MODES[0]))
        mode = str(_first(context_mode, CONTEXT_MODES[0]) or CONTEXT_MODES[0])
        ctx = snap_context(mode, int(_first(context_frames, 39)))
        if ctx != int(_first(context_frames, 39)):
            print(f"ℹ️ H3 Film Chain Render: context_frames {int(_first(context_frames, 39))} -> {ctx} ({mode.split(' (')[0]}).")
        feather = int(_first(audio_feather_ticks, 8))
        keep_latents = bool(_first(save_latents, True))
        xfade_ms = float(_first(seam_crossfade_ms, 400) or 0)
        track, track_sr = None, None
        scenes, lengths = _as_list(scenes), [int(x) for x in _as_list(lengths)]
        n = min(len(scenes), len(lengths))
        if n == 0:
            raise ValueError("H3 Film Chain Render: no scenes - wire the writer's scenes / lengths.")

        ref_images = {}
        for k in range(1, MAX_REFS + 1):
            img = _first(refs.get(f"ref_image_{k}"))
            if img is not None:
                ref_images[f"ref_image_{len(ref_images)}"] = img

        flow_all = continuity.startswith("flow")
        ref2va = _node("MiniMaxH3ReferenceToVideo")
        guider_cls, noise_cls = _node("BasicGuider"), _node("RandomNoise")
        sampler_cls, decode_cls, decode_audio_cls = _node("SamplerCustomAdvanced"), _node("VAEDecode"), _node("VAEDecodeAudio")
        import nodes as comfy_nodes
        av_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3GeneratedAVMaskedContext")
        mc_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3MotionContext")
        guide_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3AddGuide")
        # the chosen mechanism, falling back down the list when a pack node is missing
        if mode.startswith("masked") and av_cls is not None:
            mechanism = "masked AV latent (picture + sound)"
        elif (mode.startswith("masked") or mode.startswith("motion")) and mc_cls is not None:
            mechanism = "motion context guide (picture + sound)"
        elif guide_cls is not None:
            mechanism = "core guide clip (picture only)"
        else:
            mechanism = "none"
        if flow_all and not mechanism.startswith(mode.split(" ")[0]):
            print(f"⚠️ H3 Film Chain Render: '{mode.split(' (')[0]}' needs ComfyUI-H3-Motion-Context-MultiRef - using {mechanism}.")

        paths, lines = [], []
        lines.append(f"FILM CHAIN RENDER | {n} scene(s) | context {ctx} frames (~{ctx / fps_v:.2f} s) | "
                     f"continuity: {continuity.split(' (')[0]} | mechanism: {mechanism}")
        prev_latent = None
        prev_frames = None
        for i in range(n):
            length = lengths[i]
            flow = flow_all and prev_latent is not None and mechanism != "none"
            if flow and grid_up(length + ctx) > MAX_FRAMES:
                print(f"⚠️ H3 Film Chain Render: scene {i + 1:02d} is {length} frames - with {ctx} pinned frames the "
                      f"render would exceed H3's {MAX_FRAMES}-frame range, so it opens on a hard cut instead. "
                      "Shorter scenes (or fewer context frames) let it continue.")
                flow = False
            render_len = grid_up(length + ctx) if flow else grid_up(length)
            print(f"🎞️ H3 Film Chain Render | scene {i + 1:02d}/{n} | {'CONTINUES' if flow else 'hard cut'} | "
                  f"{length} frames{f' (+{ctx} pinned, rendered as {render_len})' if flow else ''}")

            cond, latent = _call(ref2va, clip=clip, vae=vae, audio_vae=audio_vae, prompt=scenes[i],
                                 width=width, height=height, length=render_len, ref_image_size="match",
                                 ref_images=ref_images or None)[:2]
            trim = 0
            if flow and mechanism.startswith("masked"):
                latent, trim = _call(av_cls, latent=latent, source_latent=prev_latent,
                                     context_length=ctx, audio_feather_ticks=feather)[:2]
                trim = int(trim)
            elif flow and mechanism.startswith("motion"):
                # previous frames as never-denoised guide rows at target frame 0, the
                # previous sound's tail pinned on the same timeline from its latent
                cond, trim = _call(mc_cls, conditioning=cond, vae=vae, latent=latent, context_frames=prev_frames,
                                   context_length=ctx, encode_mode="video", anchor_mode="head", crop="disabled",
                                   audio_context_length=0, audio_mode="timeline", context_latent=prev_latent)[:2]
                trim = int(trim)
            elif flow and guide_cls is not None:
                # picture only: the previous tail as a guide clip; decode it for the guide
                prev_images = _call(decode_cls, vae=vae, samples=prev_latent)[0]
                if prev_images.dim() == 5:
                    prev_images = prev_images.reshape(-1, prev_images.shape[-3], prev_images.shape[-2], prev_images.shape[-1])
                cond = _call(guide_cls, positive=cond, latent=latent, frame_idx=0, vae=vae, image=prev_images[-ctx:])[0]
                trim = ctx
                del prev_images

            guider = _call(guider_cls, model=model, conditioning=cond)[0]
            noise = _call(noise_cls, noise_seed=seed + i)[0]
            sampled = _call(sampler_cls, noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)[0]
            images = _call(decode_cls, vae=vae, samples=sampled)[0]
            if images.dim() == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            audio = _call(decode_audio_cls, vae=audio_vae, samples=sampled)[0]

            # The delivered clip is everything after the pinned head - the scene's
            # own frames plus the grid padding. The padding stays IN the clip on
            # purpose: the next scene continues from the latent's real tail, and a
            # tail the film never showed would be a jump at the join.
            total = int(images.shape[0])
            delivered = images[trim:] if trim else images
            full = audio
            audio = trim_audio(audio, trim, int(delivered.shape[0]), fps_v)
            path = save_frames(delivered, audio, f"{prefix}_s{i + 1:02d}", fps_v, "mp4")
            # the continuous track: this scene's audio joined across the seam
            sr = int(full["sample_rate"])
            head_s = int(round(trim / fps_v * sr))
            piece = full["waveform"].detach().to("cpu")[..., : head_s + int(round(int(delivered.shape[0]) / fps_v * sr))]
            track = seam_join(track, piece, head_s if flow else 0, int(round(xfade_ms / 1000.0 * sr)), sr)
            track_sr = sr
            paths.append(path)
            extra = int(delivered.shape[0]) - length
            lines.append(f"{i + 1:02d}  {'continues' if flow else 'cut      '}  {delivered.shape[0]} frames"
                         f"{f' (+{extra} grid pad)' if extra > 0 else ''}  "
                         f"{'head ' + str(trim) + ' trimmed' if flow else ''}  -> {path}")

            # keep the sampled AV latent (and the delivered tail) for the next scene; free everything else
            prev_latent = {"samples": sampled["samples"]}
            prev_frames = delivered[-min(ctx, int(delivered.shape[0])):].detach().clone()
            if keep_latents:
                save_scene_latent(project_key(prefix), i + 1, sampled)
            del images, delivered, audio, latent, cond, guider, noise, sampled
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            mm.soft_empty_cache()

        joined = {"waveform": track, "sample_rate": int(track_sr or 44100)} if track is not None else {"waveform": None, "sample_rate": 44100}
        if track is not None and paths:
            wav_path = save_track(track, track_sr, paths[0], prefix)
            if wav_path:
                lines.append(f"audio: one continuous track, {track.shape[-1] / track_sr:.2f} s, seams crossfaded {xfade_ms:.0f} ms -> {wav_path}")
        report = "\n".join(lines)
        print(report)
        return {"ui": {"text": [report]}, "result": (paths, report, joined)}
