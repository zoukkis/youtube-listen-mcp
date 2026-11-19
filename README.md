# youtube-listen-mcp

An [MCP](https://modelcontextprotocol.io) server that lets AI assistants like Claude "listen" to YouTube videos. It can read a video's captions, or transcribe the audio locally with Whisper, and either one can be limited to a specific time slot.

## Tools

| Tool | What it does |
|---|---|
| `video_info(url)` | Title, channel, duration, chapters |
| `list_transcripts(url)` | Caption languages available |
| `get_transcript(url, start?, end?, languages?)` | YouTube captions with timestamps. Fast, no GPU |
| `transcribe(url, start?, end?, model?, language?)` | Downloads audio (only the slice you ask for) and runs Whisper locally |

Times can be given as `90`, `1:30`, `01:02:03` or `1h2m3s`.

## Install

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and `ffmpeg` for `transcribe`:

```bash
sudo apt install ffmpeg
```

```bash
git clone https://github.com/<you>/youtube-listen-mcp
cd youtube-listen-mcp
uv sync --extra whisper     # drop --extra whisper if you only want captions
```

Test it with the MCP inspector:

```bash
npx @modelcontextprotocol/inspector uv run youtube-listen-mcp
```

## Use with Claude

**Claude Code**

```bash
claude mcp add youtube-listen -- uv --directory /path/to/youtube-listen-mcp run youtube-listen-mcp
```

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
"args": ["--from", "youtube-listen-mcp[whisper] @ git+https://github.com/<you>/youtube-listen-mcp", "youtube-listen-mcp"]
```

with `"command": "uvx"`.

Then ask: *"What does the speaker say between 12:00 and 15:30 in https://youtu.be/…?"*

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `WHISPER_DEVICE` | `auto` | `cuda` or `cpu` |
| `WHISPER_COMPUTE_TYPE` | `default` | `float16` on modern NVIDIA cards; `int8` or `float32` on Pascal (e.g. Tesla P40) or CPU |

## Notes

- This gives the model text, not sound, so it gets the speech but not tone, music or other sounds.
- YouTube sometimes rate-limits or blocks caption requests, especially from cloud IPs. If that happens, `transcribe` is the fallback.
- Respect YouTube's Terms of Service and creators' rights when downloading audio.

## License

MIT
