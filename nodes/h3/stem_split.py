# APNext H3 Stem Split
#
# Splits the song into stems with Demucs htdemucs (the same separator the
# stemkit app wraps) so the beat/event analysis can listen to just the rhythm
# section. On a compressed master the 20-250 Hz spectral flux the bass-hit
# detector reads is polluted by vocals, pads and bass NOTE changes - Sound
# Events' own header notes a bassline move can fake a kick. Feed `beats_mix`
# (default drums + bass) into H3 Sound Events / H3 Beat Grid instead of the
# full song and the pulse is measured from the instruments that actually
# carry it. The `vocals` stem is also the right thing to wire into the Music
# Video Writer's `vocals` input (cleaner Whisper transcription) and H3 Masked
# Song Latent's `voice` gate.
#
# Demucs is an optional dependency - install it into ComfyUI's python:
#     python_embeded\python.exe -m pip install demucs
# The model weights (~320 MB for htdemucs) download to the torch hub cache on
# first use. Separation runs once per song and is cached by ComfyUI like any
# node output; the model is moved off the GPU afterwards so the render gets
# its VRAM back.

import torch

from ...utils.constants import CUSTOM_CATEGORY

MODELS = [
    "htdemucs (4 stems, fast)",
    "htdemucs_ft (fine-tuned - 4x slower, a little cleaner)",
]

STEM_NAMES = ("drums", "bass", "other", "vocals")


def separate_stems(audio, model):
    """
    Demucs separation shared by the node and the Sound Events preview route.

    `audio` is a ComfyUI AUDIO dict, `model` a MODELS entry or bare model name.
    Returns ({stem: [2,T] tensor at the input sample rate}, model_name, device,
    seconds). Raises RuntimeError with install instructions when demucs is
    missing.
    """
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError:
        raise RuntimeError(
            "H3 Stem Split needs the `demucs` package. Install it into ComfyUI's "
            "python - portable install:  python_embeded\\python.exe -m pip install demucs "
            "- then restart ComfyUI. The htdemucs weights (~320 MB) download on first use."
        )
    import torchaudio

    wave = audio["waveform"]
    sr = int(audio["sample_rate"])
    w = wave[0].float().cpu()
    if w.shape[0] == 1:
        w = w.repeat(2, 1)
    elif w.shape[0] > 2:
        w = w[:2]

    name = str(model).split(" ")[0]
    net = get_model(name)
    model_sr = int(net.samplerate)
    x = torchaudio.functional.resample(w, sr, model_sr) if sr != model_sr else w

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seconds = w.shape[1] / sr
    print(f"🎚️ H3 Stem Split | separating {seconds:.1f}s with {name} on {device} ...")
    # the canonical demucs normalisation: zero-mean unit-std on the mono ref
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(
            net, ((x - mean) / std)[None],
            device=device, split=True, overlap=0.25, progress=False,
        )[0]
    out = out * std + mean
    if device == "cuda":
        # hand the VRAM back to the render
        net.to("cpu")
        torch.cuda.empty_cache()

    if model_sr != sr:
        out = torchaudio.functional.resample(out, model_sr, sr)
    out = out.cpu()

    return {source: out[i] for i, source in enumerate(net.sources)}, name, device, seconds


class H3StemSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "The song to split into stems.",
                }),
                "model": (MODELS, {
                    "default": MODELS[0],
                    "tooltip": (
                        "Which Demucs model separates the song. htdemucs is the standard "
                        "4-stem hybrid transformer; htdemucs_ft runs four models and "
                        "averages them - noticeably slower, slightly cleaner stems."
                    ),
                }),
                "include_drums": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Put the drum stem into beats_mix - the kick and snare ARE the pulse.",
                }),
                "include_bass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Put the bass stem into beats_mix - bass hits ride the kick on most masters.",
                }),
                "include_vocals": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Put the vocal stem into beats_mix. Usually off: vocal energy in the "
                        "low band is exactly what fakes beats on the full mix."
                    ),
                }),
                "include_other": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Put the other stem (guitars, keys, synths, pads) into beats_mix. "
                        "On for guitar-driven tracks whose rhythm lives outside the drums."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("beats_mix", "vocals", "drums", "bass", "other", "info")
    OUTPUT_TOOLTIPS = (
        "The selected stems summed - wire into H3 Sound Events / H3 Beat Grid instead of "
        "the full song, so beats are detected from the instruments that carry them.",
        "The vocal stem - wire into the Music Video Writer's `vocals` (cleaner Whisper "
        "transcription) or H3 Masked Song Latent's voice gate.",
        "The drum stem.",
        "The bass stem.",
        "Everything else - guitars, keys, synths, pads.",
        "Model, device, length and what went into beats_mix.",
    )
    FUNCTION = "split"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Splits the song into vocals / drums / bass / other with Demucs, and sums the "
        "stems you pick into beats_mix for H3 Sound Events and H3 Beat Grid - beat "
        "detection from the rhythm section only, instead of a full mix where vocals and "
        "bass-note changes fake hits. Needs `pip install demucs` (one ~320 MB weight "
        "download on first use)."
    )

    def split(self, audio, model, include_drums=True, include_bass=True,
              include_vocals=False, include_other=False):
        sr = int(audio["sample_rate"])
        stems, name, device, seconds = separate_stems(audio, model)

        def pack(tensor):
            return {"waveform": tensor[None].contiguous(), "sample_rate": sr}

        picks = [
            ("drums", include_drums),
            ("bass", include_bass),
            ("vocals", include_vocals),
            ("other", include_other),
        ]
        chosen = [k for k, on in picks if on and k in stems]
        if not chosen:
            chosen = [k for k in ("drums", "bass") if k in stems]
            print("⚠️ H3 Stem Split: no stems selected for beats_mix - using drums + bass.")
        mix = stems[chosen[0]].clone()
        for k in chosen[1:]:
            mix += stems[k]

        info = f"{name} on {device} | {seconds:.1f}s at {sr} Hz | beats_mix = {' + '.join(chosen)}"
        print(f"🥁 H3 Stem Split | {info}")
        return (
            pack(mix),
            pack(stems["vocals"]),
            pack(stems["drums"]),
            pack(stems["bass"]),
            pack(stems["other"]),
            info,
        )
