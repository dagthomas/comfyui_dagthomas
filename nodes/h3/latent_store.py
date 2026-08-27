# APNext H3 - sampled AV latents on disk, one per rendered scene
#
# The chain renders carry each scene's sampled latent into the next scene in
# memory. A retake of one scene later needs the SAME thing - the previous
# scene's latent tail - and by then the run is over. So every chain-rendered
# scene (and every retake) also writes its sampled latent here, keyed by the
# project and the scene number:
#
#   output/apnext_latents/<project>_s07.pt   -> {"video": [1,C,T,H,W], "audio": [1,C,2,T]}
#
# and H3 Scene Retake pins the previous scene's file to the head of its own
# render, exactly as the chain render would have.

import os
import re


def latents_dir():
    import folder_paths
    d = os.path.join(folder_paths.get_output_directory(), "apnext_latents")
    os.makedirs(d, exist_ok=True)
    return d


def project_key(name):
    """
    One key for a project however it is named: the writer's raw project_name
    ("ChromeVertigoGaffer"), the prefix it hands the savers ("video/ChromeVertigoGaffer"),
    or a bundle's slug ("chromevertigogaffer") all map to the same key.
    """
    base = str(name or "").replace("\\", "/").rstrip("/").split("/")[-1]
    key = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    return key or "untitled"


def latent_path(project, scene_no):
    return os.path.join(latents_dir(), f"{project_key(project)}_s{int(scene_no):02d}.pt")


def save_scene_latent(project, scene_no, latent):
    """Write a sampled H3 AV latent (the sampler's output dict) for `scene_no`; returns the path or None."""
    import torch
    try:
        samples = latent["samples"]
        parts = list(samples.unbind()) if hasattr(samples, "unbind") else list(samples)
        video, audio = parts[0], parts[1]
        path = latent_path(project, scene_no)
        torch.save({"video": video.detach().to("cpu"), "audio": audio.detach().to("cpu")}, path)
        return path
    except Exception as exc:  # a failed save must never fail the render
        print(f"⚠️ H3: could not save scene {int(scene_no):02d}'s latent: {exc}")
        return None


def load_scene_latent(project, scene_no):
    """The saved latent for `scene_no` as a sampler-style dict, or None when there is none."""
    import torch
    import comfy.nested_tensor
    path = latent_path(project, scene_no)
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu")
    return {"samples": comfy.nested_tensor.NestedTensor((data["video"], data["audio"]))}
