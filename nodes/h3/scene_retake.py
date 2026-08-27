# APNext H3 Scene Retake - render ONE scene again, after the run is done
#
# A finished video has one scene that did not land: a face that drifted, a
# move the model ignored, a cut that reads wrong. Re-running the graph writes
# the whole film again (a randomising writer seed) and renders every scene
# (twenty minutes to get to the one you wanted). This node instead reads the
# run's saved scene bundle (output/apnext_scenes/, written by every writer's
# `save_scenes`), takes ONE scene number, and renders just that scene with a
# new seed - and, if you like, a rewritten prompt.
#
# Chain continuity survives the retake: the chain renders (and this node)
# write every rendered scene's sampled latent to output/apnext_latents/
# <project>_sNN.pt, so a retake of scene 07 pins scene 06's real tail to its
# head - picture and sound - exactly as the chain render did; the retaken 07
# is then saved as the new tail for a later retake of 08.
#
# Its button queues ONLY this node (ComfyUI partial execution), so nothing
# upstream - least of all the writer - runs again.

import json
import os

from ...utils.constants import CUSTOM_CATEGORY
from .chain_render import MAX_REFS, _as_list, _call, _first, _node, grid_up, snap_run
from .clip_save import save_frames
from .film_chain_render import CONTEXT_MODES, snap_context, trim_audio
from .latent_store import latent_path, load_scene_latent, project_key, save_scene_latent
from .music_support import FPS, MAX_FRAMES, slice_audio
from .scenes_store import _KIND, _list_saved, scenes_dir

LATEST = "latest (newest bundle)"
RETAKE_CONTINUITY = [
    "continue from the previous scene's saved take when one exists",
    "hard cut (independent clip)",
]


def _load_bundle(choice):
    files = _list_saved()
    if not files:
        raise ValueError("No saved scenes yet. Run a writer with `save_scenes` on first - bundles land in output/apnext_scenes/.")
    name = files[0] if (not choice or choice == LATEST) else os.path.basename(str(choice))
    path = os.path.join(scenes_dir(), name)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("kind") != _KIND:
        raise ValueError(f"{name} is not an APNext H3 scenes bundle.")
    return name, data


class H3SceneRetake:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "master_audio": ("AUDIO", {
                "tooltip": (
                    "Music videos: the whole song (Load Audio). The scene's piece is cut from it at the "
                    "bundle's segment and masked into the audio latent, as the render did. Leave it "
                    "unconnected for a film - H3 then generates the sound (continuing the previous "
                    "scene's when the take continues)."
                ),
            }),
        }
        for k in range(1, MAX_REFS + 1):
            optional[f"ref_image_{k}"] = ("IMAGE", {"tooltip": f"Reference picture <Picture {k}> (Ref2VA) - the same ones the run used."})
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "H3 audio VAE."}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "bundle": ([LATEST] + _list_saved(), {
                    "default": LATEST,
                    "tooltip": (
                        "The run's saved scenes (output/apnext_scenes/, newest first). `latest` always "
                        "means the newest bundle, so a fresh run needs no browser refresh."
                    ),
                }),
                "scene_number": ("INT", {"default": 1, "min": 1, "max": 99, "tooltip": "Which scene to render again (1-based, as the writer numbers them)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "A new seed = a new take. Fixed by default so a click is one retake."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "prompt_override": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "Empty: the scene's prompt from the bundle, unchanged. Anything here replaces it "
                        "for this take - paste the scene from the Prompt Preview, edit, retake."
                    ),
                }),
                "continuity": (RETAKE_CONTINUITY, {"default": RETAKE_CONTINUITY[0]}),
                "context_frames": ("INT", {
                    "default": 39, "min": 5, "max": 141, "step": 1,
                    "tooltip": "Pinned run from the previous scene's saved take. Use what the chain render used; snapped to what context_mode can pin (39 / 90 / 141 for the masked AV latent, any H3 run for the guides).",
                }),
                "audio_feather_ticks": ("INT", {"default": 8, "min": 0, "max": 64, "tooltip": "Release over the pinned sound's last ticks (40 = 1 s); films only."}),
                "filename_prefix": ("STRING", {"default": "video/H3_retake", "tooltip": "Where the take lands; the scene number and `_retake` are appended."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.01}),
                "context_mode": (CONTEXT_MODES, {
                    "default": CONTEXT_MODES[0],
                    "tooltip": "How the previous take reaches this one - use what the chain render used (films). Music videos always use the masked song path.",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("file_path", "prompt", "report")
    OUTPUT_NODE = True
    FUNCTION = "retake"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Render one scene of a finished run again - from the saved scene bundle, with a new seed and "
        "an optional rewritten prompt, continuing from the previous scene's saved take where the chain "
        "render did. The button queues only this node, so the writer never runs again."
    )

    @classmethod
    def IS_CHANGED(cls, bundle=LATEST, **kwargs):
        # the newest bundle may change under `latest`; everything else is in the inputs
        files = _list_saved()
        name = files[0] if (not bundle or bundle == LATEST) else bundle
        try:
            return os.path.getmtime(os.path.join(scenes_dir(), name))
        except Exception:
            return float("nan")

    def retake(self, model, clip, vae, audio_vae, sampler, sigmas, bundle, scene_number, seed, width, height,
               prompt_override, continuity, context_frames, audio_feather_ticks, filename_prefix, fps,
               master_audio=None, context_mode=None, **refs):
        import torch
        import comfy.model_management as mm

        name, data = _load_bundle(bundle)
        scenes = [str(s) for s in data.get("scenes", [])]
        lengths = [int(x) for x in data.get("lengths", [])]
        segments = [tuple(s) for s in data.get("segments", [])]
        project = data.get("project_name") or os.path.splitext(name)[0]
        n = int(scene_number)
        if not scenes:
            raise ValueError(f"{name} contains no scenes.")
        if n < 1 or n > len(scenes):
            raise ValueError(f"{name} has {len(scenes)} scene(s); scene_number {n} is out of range.")
        prompt = (prompt_override or "").strip() or scenes[n - 1]
        length = lengths[n - 1] if n - 1 < len(lengths) else grid_up(int(round(float(data.get("durations", [8.0])[n - 1]) * FPS)))
        seg = segments[n - 1] if n - 1 < len(segments) else None
        fps_v = float(fps or FPS)
        mode = str(context_mode or CONTEXT_MODES[0])
        if master_audio is not None:
            # the masked song path: the whole audio stream is the song, so only the
            # video prefix has to be a valid run - any 5 + 17k, as the chain render used
            mode = CONTEXT_MODES[0]
            ctx = max(5, snap_run(int(context_frames)))
        else:
            ctx = snap_context(mode, int(context_frames))
        if ctx != int(context_frames):
            print(f"ℹ️ H3 Scene Retake: context_frames {int(context_frames)} -> {ctx}.")
        prefix = str(filename_prefix or "video/H3_retake")

        ref_images = {}
        for k in range(1, MAX_REFS + 1):
            img = refs.get(f"ref_image_{k}")
            if img is not None:
                ref_images[f"ref_image_{len(ref_images)}"] = img

        prev_latent = None
        if str(continuity).startswith("continue") and n > 1:
            prev_latent = load_scene_latent(project, n - 1)
            if prev_latent is None:
                print(f"ℹ️ H3 Scene Retake: no saved take for scene {n - 1:02d} at {latent_path(project, n - 1)} - "
                      "rendering scene {n:02d} as a hard cut. Chain renders save one per scene when `save_latents` is on.")
        flow = prev_latent is not None
        if flow and grid_up(length + ctx) > MAX_FRAMES:
            print(f"⚠️ H3 Scene Retake: scene {n:02d} is {length} frames - with {ctx} pinned frames the render would exceed "
                  f"H3's {MAX_FRAMES}-frame range, so it opens on a hard cut instead.")
            flow = False

        import nodes as comfy_nodes
        ref2va = _node("MiniMaxH3ReferenceToVideo")
        guider_cls, noise_cls = _node("BasicGuider"), _node("RandomNoise")
        sampler_cls, decode_cls, decode_audio_cls = _node("SamplerCustomAdvanced"), _node("VAEDecode"), _node("VAEDecodeAudio")
        masked_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3SongMaskedAVContext")
        av_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3GeneratedAVMaskedContext")
        mc_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3MotionContext")
        guide_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3AddGuide")
        use_av = mode.startswith("masked") and av_cls is not None
        use_mc = not use_av and (mode.startswith("masked") or mode.startswith("motion")) and mc_cls is not None
        if master_audio is not None and masked_cls is None:
            print("⚠️ H3 Scene Retake: master_audio is connected but ComfyUI-H3-Motion-Context-MultiRef is not installed - "
                  "the song cannot be masked in; H3 will generate the sound.")
        if flow and master_audio is None and not use_av and not use_mc and guide_cls is None:
            flow = False

        render_len = grid_up(length + ctx) if flow else grid_up(length)
        head = ctx if flow else 0
        start = (float(seg[0]) if seg else 0.0) - head / fps_v
        mech = ("masked song + previous take" if (master_audio is not None and masked_cls) else
                "previous take (masked AV latent)" if use_av else
                "previous take (motion context guide)" if use_mc else "previous take (picture)") if flow else \
               ("masked song" if (master_audio is not None and masked_cls) else "plain")
        print(f"🎬 H3 Scene Retake | {name} | scene {n:02d}/{len(scenes)} | {'CONTINUES' if flow else 'hard cut'} | "
              f"{length} frames{f' (+{head} pinned, rendered as {render_len})' if flow else ''} | seed {int(seed)} | {mech}"
              + (" | prompt override" if (prompt_override or '').strip() else ""))

        cond, latent = _call(ref2va, clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt, width=int(width), height=int(height),
                             length=render_len, ref_image_size="match", ref_images=ref_images or None)[:2]
        trim = 0
        if master_audio is not None and masked_cls is not None:
            if flow:
                latent, trim, _clip = _call(masked_cls, latent=latent, audio_vae=audio_vae, master_audio=master_audio,
                                            clip_start_seconds=max(0.0, start), context_length=ctx, source_fps=fps_v, crop="disabled",
                                            source_latent=prev_latent)[:3]
                trim = int(trim)
            else:
                latent = _call(masked_cls, latent=latent, audio_vae=audio_vae, master_audio=master_audio,
                               clip_start_seconds=max(0.0, start), context_length=0, source_fps=fps_v, crop="disabled")[0]
        elif flow and use_av:
            latent, trim = _call(av_cls, latent=latent, source_latent=prev_latent, context_length=ctx,
                                 audio_feather_ticks=int(audio_feather_ticks))[:2]
            trim = int(trim)
        elif flow and use_mc:
            prev_images = _call(decode_cls, vae=vae, samples=prev_latent)[0]
            if prev_images.dim() == 5:
                prev_images = prev_images.reshape(-1, prev_images.shape[-3], prev_images.shape[-2], prev_images.shape[-1])
            cond, trim = _call(mc_cls, conditioning=cond, vae=vae, latent=latent, context_frames=prev_images[-ctx:],
                               context_length=ctx, encode_mode="video", anchor_mode="head", crop="disabled",
                               audio_context_length=0, audio_mode="timeline", context_latent=prev_latent)[:2]
            trim = int(trim)
            del prev_images
        elif flow and guide_cls is not None:
            prev_images = _call(decode_cls, vae=vae, samples=prev_latent)[0]
            if prev_images.dim() == 5:
                prev_images = prev_images.reshape(-1, prev_images.shape[-3], prev_images.shape[-2], prev_images.shape[-1])
            cond = _call(guide_cls, positive=cond, latent=latent, frame_idx=0, vae=vae, image=prev_images[-ctx:])[0]
            trim = ctx
            del prev_images

        guider = _call(guider_cls, model=model, conditioning=cond)[0]
        noise = _call(noise_cls, noise_seed=int(seed))[0]
        sampled = _call(sampler_cls, noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)[0]
        images = _call(decode_cls, vae=vae, samples=sampled)[0]
        if images.dim() == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        if master_audio is not None and seg is not None:
            # a music-video piece: exactly the scene's frames, the song piece as its sound
            delivered = images[trim:trim + length]
            audio = slice_audio(master_audio, float(seg[0]), float(seg[1]), frames=int(delivered.shape[0]))
        else:
            # a film scene: everything after the pinned head, its own decoded sound trimmed alike
            delivered = images[trim:] if trim else images
            audio = trim_audio(_call(decode_audio_cls, vae=audio_vae, samples=sampled)[0], trim, int(delivered.shape[0]), fps_v)

        path = save_frames(delivered, audio, f"{prefix}_s{n:02d}_retake", fps_v, "mp4")
        saved = save_scene_latent(project, n, sampled)
        report = (f"RETAKE | {name} | scene {n:02d} | {'continues from ' + f'scene {n - 1:02d}' if flow else 'hard cut'} | "
                  f"{int(delivered.shape[0])} frames | seed {int(seed)} | {mech}\n-> {path}"
                  + (f"\nlatent saved for a later retake of scene {n + 1:02d}: {saved}" if saved else ""))
        print(report)
        del images, delivered, sampled, latent, cond, guider, noise
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mm.soft_empty_cache()
        return {"ui": {"text": [report]}, "result": (path, prompt, report)}
