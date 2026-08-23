# Whisper lyric transcription for the H3 Music Video Writer.
#
# When the writer gets no lyrics, it can pull them from the song itself:
# Whisper (via the transformers pipeline already shipped with this ComfyUI)
# transcribes the track into timed `[m:ss] line` rows - the exact format
# parse_lyrics() consumes, so timed cuts and lyrics-driven scenes work the
# same as with hand-typed lyrics. A vocal stem transcribes noticeably better
# than the full mix; wire one in when the workflow already separates stems.
#
# The first use downloads the model from Hugging Face (~1.6 GB for
# whisper-large-v3-turbo) into the HF cache. Transcriptions are cached
# in-process by waveform fingerprint, so re-queuing the same song is free.

import hashlib

_MODEL_ID = "openai/whisper-large-v3-turbo"
_CACHE = {}


def _fingerprint(audio):
    import torch
    wav = audio["waveform"]
    h = hashlib.sha1()
    h.update(str((tuple(wav.shape), int(audio.get("sample_rate", 0)))).encode())
    flat = wav.reshape(-1)
    step = max(1, flat.numel() // 4096)
    h.update(flat[::step].to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()


def _format_lines(chunks):
    lines, prev = [], None
    for ch in chunks or []:
        text = " ".join((ch.get("text") or "").split())
        if not text:
            continue
        # Whisper hallucinates loops on instrumental stretches - drop repeats
        if prev is not None and text.lower() == prev.lower():
            continue
        prev = text
        start = (ch.get("timestamp") or (None,))[0]
        if start is None:
            lines.append(text)
        else:
            m, s = divmod(int(start), 60)
            lines.append(f"[{m}:{s:02d}] {text}")
    return "\n".join(lines)


def transcribe_song_lyrics(audio, model_id=_MODEL_ID):
    """Timed `[m:ss] line` lyrics from an AUDIO dict, or "" on failure."""
    try:
        key = _fingerprint(audio)
        if key in _CACHE:
            return _CACHE[key]

        import torch
        import torchaudio
        from transformers import pipeline

        wav = audio["waveform"][:1].mean(dim=1)          # [1, N] mono
        sr = int(audio["sample_rate"])
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        cuda = torch.cuda.is_available()
        asr = pipeline(
            "automatic-speech-recognition", model=model_id,
            torch_dtype=torch.float16 if cuda else torch.float32,
            device=0 if cuda else -1,
        )
        try:
            out = asr(
                {"array": wav[0].to(torch.float32).cpu().numpy(), "sampling_rate": 16000},
                return_timestamps=True, chunk_length_s=30,
            )
        finally:
            del asr                    # a song render follows - give the VRAM back
            if cuda:
                torch.cuda.empty_cache()

        result = _format_lines(out.get("chunks"))
        if not result and (out.get("text") or "").strip():
            result = " ".join(out["text"].split())
        _CACHE[key] = result
        return result
    except Exception as exc:
        print(f"⚠️ H3 lyric transcription failed ({type(exc).__name__}): {exc}")
        return ""
