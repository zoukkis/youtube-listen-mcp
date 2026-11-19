# youtube-listen-mcp

An [MCP](https://modelcontextprotocol.io) server that lets AI assistants like Claude "listen" to YouTube videos. It can read a video's captions, or transcribe the audio locally with Whisper, and either one can be limited to a specific time slot.

## Tools

| Tool | What it does |
|---|---|
| `video_info(url)` | Title, channel, duration, chapters |
| `list_transcripts(url)` | Caption languages available |
| `get_transcript(url, start?, end?, languages?)` | YouTube captions with timestamps. Fast, no GPU |
| `transcribe(url, start?, end?, model?, language?)` | Downloads audio (only the slice you ask for) and runs Whisper locally |
| `screenshots(url, times, max_height?)` | Frames at the given timestamps (up to 10), returned as images the model can see. Only those frames are fetched, not the whole video. Up to 720p by default |

Times can be given as `90`, `1:30`, `01:02:03` or `1h2m3s`.

## Install

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and `ffmpeg` for `transcribe` and `screenshots`:

```bash
sudo apt install ffmpeg
```

```bash
git clone https://github.com/zoukkis/youtube-listen-mcp
cd youtube-listen-mcp
uv sync --extra whisper     # drop --extra whisper if you only want captions
```

Test it with the MCP inspector:

```bash
npx @modelcontextprotocol/inspector uv run youtube-listen-mcp
```

### GPU (NVIDIA)

On Linux, `--extra whisper` also installs NVIDIA's CUDA 12 cuBLAS and cuDNN 9 libraries from PyPI (about 1 GB), and the server loads them automatically. You don't need a system-wide CUDA toolkit, `sudo` or `LD_LIBRARY_PATH`. You only need an NVIDIA driver that supports CUDA 12 or later; `nvidia-smi` shows the version in its top right corner.

To force the GPU and use half precision, set:

```bash
WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16
```

- **No NVIDIA GPU, or on macOS:** use `WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=int8`. The `small` model is fine on CPU for short clips.
- **Windows:** the bundled libraries are Linux only. Install CUDA 12 cuBLAS and cuDNN 9 yourself and put them on `PATH`, or use CPU.
- **`Library libcublas.so.12 is not found`:** the extra isn't installed. Run `uv sync --extra whisper`.
- **`open() got an unexpected keyword argument 'metadata_errors'`:** PyAV 19 or later is installed. The extra pins `av<19`, so running `uv sync --extra whisper` again fixes it.

## Use with Claude

**Claude Code**

```bash
claude mcp add -s user youtube-listen -- uv --directory /path/to/youtube-listen-mcp run youtube-listen-mcp
```

`-s user` makes it available in every project. Add `-e WHISPER_DEVICE=cuda -e WHISPER_COMPUTE_TYPE=float16` before `--` to use the GPU.

**Claude Desktop** (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "youtube-listen": {
      "command": "uv",
      "args": ["--directory", "/path/to/youtube-listen-mcp", "run", "youtube-listen-mcp"],
      "env": { "WHISPER_MODEL": "large-v3" }
    }
  }
}
```

Or run it straight from GitHub without cloning:

```json
"args": ["--from", "youtube-listen-mcp[whisper] @ git+https://github.com/zoukkis/youtube-listen-mcp", "youtube-listen-mcp"]
```

with `"command": "uvx"`.

Then ask: *"What does the speaker say between 12:00 and 15:30 in https://youtu.be/…?"*

Or combine speech and visuals: *"In https://youtu.be/…, transcribe 19:12–21:22 and look at the screen at 17:11 and 19:21. What's being shown and explained?"*

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `WHISPER_DEVICE` | `auto` | `cuda` or `cpu` |
| `WHISPER_COMPUTE_TYPE` | `default` | `float16` on modern NVIDIA cards; `int8` or `float32` on Pascal (e.g. Tesla P40) or CPU |

## Notes

- This gives the model text, not sound, so it gets the speech but not tone, music or other sounds. `screenshots` adds still frames, not motion.
- YouTube sometimes rate-limits or blocks caption requests, especially from cloud IPs. If that happens, `transcribe` is the fallback.
- Respect YouTube's Terms of Service and creators' rights when downloading audio.

## License

MIT
