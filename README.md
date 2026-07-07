# Comfyui-ZhiXiapi

ComfyUI custom node package for **ZhiXiapi** (zaiduyu.top), providing GPT Image 2 generation nodes.

## Requirements

- **Python**: 3.10+

## Dependencies

| Library | Version | Notes |
|---------|---------|-------|
| `requests` | >=2.28 | HTTP client |
| `numpy` | >=1.20 | Array processing (ComfyUI built-in) |
| `Pillow` | >=9.0 | Image processing (ComfyUI built-in) |
| `torch` | >=2.0 | Tensor processing (ComfyUI built-in) |

Most dependencies are bundled with ComfyUI; no extra install usually needed.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Tompulsen/ComfyUI-zaiduyun.git Comfyui-ZhiXiapi
cd Comfyui-ZhiXiapi
pip install -r requirements.txt
```

Restart ComfyUI and search for `zhixiapi` or `zaiduyun`.

## Nodes

| Node | Description |
|------|-------------|
| **Zhixiapi-image-2** | GPT Image 2 node: text2img, img2img, multi-image edit, mask support |

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| `api_key` | API key from zaiduyu.top |
| `api_base` | API endpoint (default: `https://zaiduyu.top`) |
| `model` | Model name (default: `gpt-image-2`) |
| `timeout_seconds` | Request timeout (60-1800s) |
| `retry_times` | Retry count (1-10) |

## API Endpoints

| Endpoint | Usage |
|----------|-------|
| `POST /v1/images/generations` | Text-to-image |
| `POST /v1/images/edits` | Image-to-image / inpainting |

## Features

- Configurable API base URL with proxy fallback
- Pre-upload image compression for stability
- Adaptive retry with exponential backoff
- Frontend runtime status bar with live progress
- Multi-image reference support (up to 8 images)
- Mask-based inpainting
- Custom resolution support (1K/2K/4K with aspect ratios)
- Skip-error mode for workflow continuity

## Online Workflows

- Overseas: https://www.runninghub.cn/?inviteCode=ctafknae
- Domestic: https://www.runninghub.cn/?inviteCode=ctafknae


## License

See [LICENSE](LICENSE)
