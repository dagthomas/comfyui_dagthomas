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


from ...utils.constants import CUSTOM_CATEGORY
from .chain_render import MAX_REFS, _as_list, _call, _first, _node, grid_up
from .clip_save import save_frames
from .music_support import FPS, MAX_FRAMES

FILM_CONTINUITY_MODES = [
    "flow everywhere (every scene continues the previous picture and sound)",
    "cut everywhere (independent clips - render as before)",
]

# H3 runs where the 24 fps picture and the 40 Hz audio latent share a
# boundary: 39, 90, 141, ... frames (39 + 51k). Only these pin the sound to
# the same instant as the picture.
AV_ALIGNED_RUNS = tuple(39 + 51 * k for k in range(0, 8))


def snap_av_run(n):
    """The largest AV-aligned context run (39, 90, 141, ...) not above n; 39 at least."""
    n = int(n)
    best = AV_ALIGNED_RUNS[0]
    for run in AV_ALIGNED_RUNS:
        if run <= n:
            best = run
    return best


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
                    "default": 39, "min": 39, "max": 141, "step": 1,
                    "tooltip": (
                        "How much of the previous scene - picture AND sound - is pinned to the head of the "
                        "next one. Snapped to the runs where H3's 24 fps picture and 40 Hz audio latent share "
                        "a boundary: 39 (~1.6 s), 90 (~3.75 s), 141 (~5.9 s). More context = a longer, "
                        "surer continuation of the sound, and fewer frames left for the scene itself."
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
        "Render a short film's scenes one after another, pinning the last frames AND the last sound of "
        "each scene into the head of the next (straight from the sampled latent), so the picture continues "
        "and the room tone / score / voice do not restart at every scene. Ref2VA, masked AV prefix, "
        "sample, decode, trim, save - per scene, inside one node."
    )

    def render(self, model, clip, vae, audio_vae, scenes, lengths, width, height, sampler, sigmas, seed,
               context_frames, audio_feather_ticks, continuity, filename_prefix, fps, **refs):
        import torch
        import comfy.model_management as mm

        model, clip, vae, audio_vae = _first(model), _first(clip), _first(vae), _first(audio_vae)
        sampler, sigmas = _first(sampler), _first(sigmas)
        width, height = int(_first(width, 1344)), int(_first(height, 768))
        seed = int(_first(seed, 0))
        fps_v = float(_first(fps, FPS))
        prefix = str(_first(filename_prefix, "video/H3_film"))
        continuity = str(_first(continuity, FILM_CONTINUITY_MODES[0]))
        ctx = snap_av_run(int(_first(context_frames, 39)))
        feather = int(_first(audio_feather_ticks, 8))
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
        guide_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3AddGuide")
        if flow_all and av_cls is None:
            print("⚠️ H3 Film Chain Render: ComfyUI-H3-Motion-Context-MultiRef is not installed - only the picture "
                  "can be carried (core guide clip); each scene's sound is generated afresh.")
        mechanism = "masked AV prefix (picture + sound)" if av_cls else ("core guide clip (picture only)" if guide_cls else "none")

        paths, lines = [], []
        lines.append(f"FILM CHAIN RENDER | {n} scene(s) | context {ctx} frames (~{ctx / fps_v:.2f} s) | "
                     f"continuity: {continuity.split(' (')[0]} | mechanism: {mechanism}")
        prev_latent = None
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
            if flow and av_cls is not None:
                latent, trim = _call(av_cls, latent=latent, source_latent=prev_latent,
                                     context_length=ctx, audio_feather_ticks=feather)[:2]
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
            audio = trim_audio(audio, trim, int(delivered.shape[0]), fps_v)
            path = save_frames(delivered, audio, f"{prefix}_s{i + 1:02d}", fps_v, "mp4")
            paths.append(path)
            extra = int(delivered.shape[0]) - length
            lines.append(f"{i + 1:02d}  {'continues' if flow else 'cut      '}  {delivered.shape[0]} frames"
                         f"{f' (+{extra} grid pad)' if extra > 0 else ''}  "
                         f"{'head ' + str(trim) + ' trimmed' if flow else ''}  -> {path}")

            # keep the sampled AV latent for the next scene; free everything else
            prev_latent = {"samples": sampled["samples"]}
            del images, delivered, audio, latent, cond, guider, noise, sampled
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            mm.soft_empty_cache()

        report = "\n".join(lines)
        print(report)
        return {"ui": {"text": [report]}, "result": (paths, report)}
