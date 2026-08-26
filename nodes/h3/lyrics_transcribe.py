# Whisper lyric transcription for the H3 Music Video Writer.
#
# When the writer gets no lyrics, it can pull them from the song itself:
# Whisper (via transformers, already shipped with this ComfyUI) transcribes
# the track into timed `[m:ss] line` rows - the exact format parse_lyrics()
# consumes, so timed cuts and lyrics-driven scenes work the same as with
# hand-typed lyrics. A vocal stem transcribes noticeably better than the
# full mix; wire one in when the workflow already separates stems.
#
# The model is driven directly (processor + generate) rather than through
# the `automatic-speech-recognition` pipeline: the pipeline imports torchcodec
# whenever it is installed, and on the portable Windows build that import
# dies looking for shared FFmpeg DLLs - even though we hand it a raw array
# and never decode a file. Direct generate also uses Whisper's own long-form
# algorithm instead of the pipeline's "experimental" 30 s chunking.
#
# The first use downloads the model from Hugging Face (~1.6 GB for
# whisper-large-v3-turbo) into the HF cache. Transcriptions are cached
# in-process by waveform fingerprint, so re-queuing the same song is free.

import hashlib
import re

_MODEL_ID = "openai/whisper-large-v3-turbo"
_CACHE = {}

_TIMESTAMP_TOKEN = re.compile(r"<\|(\d+\.\d+)\|>")
_ANY_SPECIAL = re.compile(r"<\|[^|]*\|>")


def _fingerprint(audio):
    import torch
    wav = audio["waveform"]
    h = hashlib.sha1()
    h.update(str((tuple(wav.shape), int(audio.get("sample_rate", 0)))).encode())
    flat = wav.reshape(-1)
    step = max(1, flat.numel() // 4096)
    h.update(flat[::step].to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()


# Captions Whisper invents for silence and instrumentals - subtitle credits it
# learned from broadcast transcripts. A song gets these in every empty window.
_HALLUCINATIONS = re.compile(
    r"^(teksting av|tekstet av|textning|undertekst\w* av|subtitles? by|subtitled by|"
    r"transcri\w+ by|amara\.org|thanks? for watching|thank you for watching|"
    r"www\.|\[music\]|\(music\)|♪)",
    re.IGNORECASE,
)
_MAX_REPEATS = 3        # a real chorus repeats; the same caption in every window is a loop
_HAS_WORD = re.compile(r"\w")


def _format_lines(segments):
    """[(start_seconds_or_None, text), ...] -> timed `[m:ss] line` rows."""
    counts = {}
    for _, text in segments or []:
        key = " ".join((text or "").split()).lower()
        counts[key] = counts.get(key, 0) + 1

    lines, prev = [], None
    for start, text in segments or []:
        text = " ".join((text or "").split())
        if not _HAS_WORD.search(text):        # empty, or only punctuation / zero-width junk
            continue
        # Whisper hallucinates loops on instrumental stretches - drop repeats
        if prev is not None and text.lower() == prev.lower():
            continue
        prev = text
        if _HALLUCINATIONS.match(text.strip("\"'“”‘’ ")):
            continue
        if counts.get(text.lower(), 0) > _MAX_REPEATS and len(text) < 40:
            continue
        if start is None:
            lines.append(text)
        else:
            m, s = divmod(int(start), 60)
            lines.append(f"[{m}:{s:02d}] {text}")
    return "\n".join(lines)


def _split_on_timestamps(decoded):
    """'<|0.00|> Hello<|2.40|><|2.40|> world<|4.00|>' -> [(0.0, 'Hello'), (2.4, 'world')]."""
    segs, cur = [], None
    for i, piece in enumerate(_TIMESTAMP_TOKEN.split(decoded)):
        if i % 2 == 1:
            cur = float(piece)
            continue
        text = _ANY_SPECIAL.sub("", piece).strip()
        if text:
            segs.append((cur, text))
    return segs


def _generate_segments(model, processor, wav16k, device, dtype):
    """
    Run Whisper over a mono 16 kHz float tensor and return
    [(start_seconds, text), ...]. One `generate` call handles a whole song:
    the model chunks it on its own timestamp tokens, which is what the
    Whisper paper describes and what the pipeline's fixed 30 s windows only
    approximate.
    """
    import torch

    long_form = wav16k.numel() > 30 * 16000
    if long_form:
        features = processor(
            wav16k.numpy(), sampling_rate=16000, return_tensors="pt",
            truncation=False, padding="longest", return_attention_mask=True,
        )
    else:
        # the encoder takes exactly 30 s: let the processor pad up to it
        features = processor(wav16k.numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = features["input_features"].to(device=device, dtype=dtype)
    kwargs = {"return_timestamps": True, "task": "transcribe"}
    attention_mask = features.get("attention_mask")
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(device)

    if long_form:
        # Whisper's own hallucination guards (the openai-whisper defaults).
        # Without them a song full of instrumental stretches comes back as
        # the same invented caption in every window - on Norwegian audio
        # famously "Teksting av Nicolai Winther".
        kwargs.update(
            return_segments=True,
            condition_on_prev_tokens=False,
            compression_ratio_threshold=1.35,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )
    with torch.inference_mode():
        out = model.generate(input_features, **kwargs)

    if long_form and isinstance(out, dict) and out.get("segments"):
        segs = []
        for seg in out["segments"][0]:
            start = seg.get("start")
            start = float(start) if start is not None else None
            text = processor.tokenizer.decode(seg["tokens"], skip_special_tokens=True)
            segs.append((start, text))
        return segs

    # A clip under 30 s (or a build without segment output): decode with the
    # timestamp tokens kept and split on them ourselves.
    ids = out["sequences"] if isinstance(out, dict) else out
    decoded = processor.tokenizer.decode(ids[0], skip_special_tokens=False, decode_with_timestamps=True)
    segs = _split_on_timestamps(decoded)
    if not segs:
        plain = processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        if plain:
            segs.append((None, plain))
    return segs


def transcribe_song_lyrics(audio, model_id=_MODEL_ID):
    """Timed `[m:ss] line` lyrics from an AUDIO dict, or "" on failure."""
    try:
        key = _fingerprint(audio)
        if key in _CACHE:
            return _CACHE[key]

        import torch
        import torchaudio
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        wav = audio["waveform"][:1].mean(dim=1)          # [1, N] mono
        sr = int(audio["sample_rate"])
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav16k = wav[0].to(torch.float32).cpu()

        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        dtype = torch.float16 if cuda else torch.float32
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device)
        model.eval()
        try:
            segments = _generate_segments(model, processor, wav16k, device, dtype)
        finally:
            del model                  # a song render follows - give the VRAM back
            if cuda:
                torch.cuda.empty_cache()

        result = _format_lines(segments)
        _CACHE[key] = result
        return result
    except Exception as exc:
        msg = f"H3 lyric transcription failed ({type(exc).__name__}): {exc}"
        try:
            print(f"⚠️ {msg}")
        except UnicodeEncodeError:       # cp1252 console
            print(f"WARNING: {msg}")
        return ""
