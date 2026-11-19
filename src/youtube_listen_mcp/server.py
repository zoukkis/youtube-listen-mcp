"""MCP server that lets an AI assistant "listen" to YouTube videos.

Tools:
  - video_info:       title, duration, chapters (to find the right time slot)
  - list_transcripts: which caption languages a video has
  - get_transcript:   YouTube captions, optionally limited to a time range
  - transcribe:       download audio (or one slice of it) and run Whisper locally
  - screenshots:      grab video frames at given timestamps, returned as images
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import Image, MCPServer

MAX_SCREENSHOTS = 10

mcp = MCPServer("youtube-listen")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_ID_PATTERNS = [
    r"(?:v=|/v/|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})",
    r"^([A-Za-z0-9_-]{11})$",
]


def video_id(url_or_id: str) -> str:
    """Extract the 11-character video ID from a URL (or accept a bare ID)."""
    s = url_or_id.strip()
    for pat in _ID_PATTERNS:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    raise ValueError(f"Could not find a YouTube video ID in: {url_or_id!r}")


def parse_time(t: Optional[str | int | float]) -> Optional[float]:
    """Accept 90, "90", "1:30", "01:02:03", "1h2m3s", "2m", "45s"."""
    if t is None or t == "":
        return None
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    if ":" in s:
        total = 0.0
        for part in s.split(":"):
            total = total * 60 + float(part)
        return total
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", s)
    if m and any(m.groups()):
        h, mi, se = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + se
    raise ValueError(f"Unrecognised time format: {t!r}")


def fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _lines(segments, start: Optional[float], end: Optional[float]) -> str:
    out = []
    for seg_start, seg_end, text in segments:
        if start is not None and seg_end < start:
            continue
        if end is not None and seg_start > end:
            break
        text = " ".join(text.split())
        if text:
            out.append(f"[{fmt(seg_start)}] {text}")
    return "\n".join(out)


_cuda_preloaded = False


def _preload_cuda_libs() -> None:
    """Load cuBLAS/cuDNN from the nvidia-* pip wheels, if installed.

    CTranslate2 dlopen()s libcublas.so.12 and libcudnn*.so.9 by name, which
    fails when CUDA isn't installed system-wide. Loading them first with
    RTLD_GLOBAL makes them visible without needing LD_LIBRARY_PATH.
    """
    global _cuda_preloaded
    if _cuda_preloaded or not sys.platform.startswith("linux"):
        return
    _cuda_preloaded = True
    import ctypes
    import importlib.util

    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue
        if not spec or not spec.submodule_search_locations:
            continue
        for loc in spec.submodule_search_locations:
            # cublasLt before cublas; libcudnn.so.9 last so its parts are loaded
            for lib in sorted((Path(loc) / "lib").glob("lib*.so*"),
                              key=lambda p: (p.name.startswith("libcublas."),
                                             p.name == "libcudnn.so.9")):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@mcp.tool()
def video_info(url: str) -> str:
    """Get a YouTube video's title, channel, duration and chapter list.

    Useful for finding the time slot to read before fetching a transcript.
    """
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    lines = [
        f"Title: {info.get('title')}",
        f"Channel: {info.get('uploader')}",
        f"Duration: {fmt(info.get('duration') or 0)}",
        f"Upload date: {info.get('upload_date')}",
    ]
    chapters = info.get("chapters") or []
    if chapters:
        lines.append("Chapters:")
        lines += [f"  [{fmt(c['start_time'])}] {c['title']}" for c in chapters]
    return "\n".join(lines)


@mcp.tool()
def list_transcripts(url: str) -> str:
    """List the caption tracks (languages, manual vs auto-generated) a video has."""
    from youtube_transcript_api import YouTubeTranscriptApi

    tl = YouTubeTranscriptApi().list(video_id(url))
    rows = [
        f"{t.language_code}\t{t.language}\t"
        f"{'auto-generated' if t.is_generated else 'manual'}"
        for t in tl
    ]
    return "\n".join(rows) or "No transcripts available."


@mcp.tool()
def get_transcript(
    url: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    languages: Optional[list[str]] = None,
) -> str:
    """Fetch a video's YouTube captions with timestamps.

    Args:
        url: YouTube URL or video ID.
        start: Optional start of the time slot, e.g. "12:30", "750", "12m30s".
        end: Optional end of the time slot.
        languages: Preferred language codes in order, default ["en"].

    Fast and needs no GPU. If the video has no captions, use `transcribe`.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(
        video_id(url), languages=languages or ["en"]
    )
    segs = [(s.start, s.start + s.duration, s.text) for s in fetched.snippets]
    text = _lines(segs, parse_time(start), parse_time(end))
    header = f"Transcript ({fetched.language_code}, " \
             f"{'auto' if fetched.is_generated else 'manual'} captions)"
    return f"{header}\n{text}" if text else "No captions in that time range."


@mcp.tool()
def transcribe(
    url: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Download the audio and transcribe it locally with Whisper.

    Slower than `get_transcript` but works when there are no captions and is
    usually more accurate. Pass start/end to download only that slice.

    Args:
        url: YouTube URL or video ID.
        start: Optional start of the time slot, e.g. "12:30".
        end: Optional end of the time slot.
        model: Whisper model (tiny, base, small, medium, large-v3).
               Defaults to $WHISPER_MODEL or "small".
        language: Optional language code, e.g. "en", "fi". Auto-detect if empty.
    """
    _preload_cuda_libs()
    import yt_dlp
    from yt_dlp.utils import download_range_func
    from faster_whisper import WhisperModel

    s, e = parse_time(start), parse_time(end)
    model_name = model or os.environ.get("WHISPER_MODEL", "small")
    device = os.environ.get("WHISPER_DEVICE", "auto")
    compute = os.environ.get("WHISPER_COMPUTE_TYPE", "default")

    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmp) / "audio.%(ext)s"),
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
            ],
        }
        if s is not None or e is not None:
            opts["download_ranges"] = download_range_func(
                None, [(s or 0, e or float("inf"))]
            )
            opts["force_keyframes_at_cuts"] = True

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        audio = next(Path(tmp).glob("audio.*"))
        print(f"Transcribing {audio.name} with {model_name}…", file=sys.stderr)
        wm = WhisperModel(model_name, device=device, compute_type=compute)
        segments, info = wm.transcribe(str(audio), language=language, vad_filter=True)

        offset = s or 0.0  # keep timestamps relative to the full video
        segs = [(seg.start + offset, seg.end + offset, seg.text) for seg in segments]

    text = _lines(segs, None, None)
    return f"Whisper {model_name} ({info.language})\n{text}" if text else "No speech detected."


@mcp.tool()
def screenshots(url: str, times: list[str], max_height: int = 720):
    """Grab still frames from a YouTube video at the given timestamps.

    Use this to see what is on screen (slides, code, diagrams, people) at
    specific moments. Combine with `get_transcript` or `transcribe` for the
    surrounding speech. Only the frames are fetched, not the whole video.

    Args:
        url: YouTube URL or video ID.
        times: Timestamps to capture, e.g. ["17:11", "19:21"]. Max 10.
        max_height: Maximum frame height in pixels (default 720).
    """
    import yt_dlp

    if not times:
        raise ValueError("Give at least one timestamp.")
    if len(times) > MAX_SCREENSHOTS:
        raise ValueError(f"At most {MAX_SCREENSHOTS} screenshots per call.")
    points = [(t, parse_time(t)) for t in times]

    h = int(max_height)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # A single video-only stream so there is one URL ffmpeg can seek in.
        "format": f"bestvideo[height<={h}][ext=mp4]/bestvideo[height<={h}]"
                  f"/best[height<={h}]/best",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    stream = info.get("url")
    if not stream:
        raise RuntimeError("Could not get a video stream URL for this video.")
    headers = "".join(f"{k}: {v}\r\n" for k, v in (info.get("http_headers") or {}).items())
    duration = info.get("duration")

    out: list = [f"Screenshots from: {info.get('title')}"]
    for label, sec in points:
        if duration and sec >= duration:
            out.append(f"[{label}] skipped: past the end of the video ({fmt(duration)})")
            continue
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if headers:
            cmd += ["-headers", headers]
        cmd += ["-ss", str(sec), "-i", stream, "-frames:v", "1",
                "-vf", f"scale=-2:'min({h},ih)'", "-q:v", "3",
                "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode(errors="replace").strip()[-300:]
            out.append(f"[{label}] failed: {err or 'no frame returned'}")
            continue
        out.append(f"[{fmt(sec)}]")
        out.append(Image(data=proc.stdout, format="jpeg"))
    return out


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
