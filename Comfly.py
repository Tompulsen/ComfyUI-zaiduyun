import base64
import json
import math
import random
import time
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_API_BASE_URL = "http://154.37.221.15:3030"
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 300
DEFAULT_NODE_TIMEOUT = 300
DEFAULT_MIN_NODE_TIMEOUT = 60
DEFAULT_MAX_NODE_TIMEOUT = 1800
DEFAULT_RETRY_TIMES = 1
FAL_SEED_MAX = 65535

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_LEVELS = {"debug": 0, "info": 1, "warn": 2, "error": 3}
_LOG_MIN_LEVEL = "info"
_LOG_MAX_LENGTH = 2000


def _log(level, *args):
    if _LOG_LEVELS.get(level, 99) < _LOG_LEVELS.get(_LOG_MIN_LEVEL, 1):
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    tag = level.upper().ljust(5)
    msg = " ".join(str(a) for a in args)
    if len(msg) > _LOG_MAX_LENGTH:
        msg = msg[:_LOG_MAX_LENGTH] + f"...<truncated {len(msg) - _LOG_MAX_LENGTH} chars>"
    print(f"[{ts}] [{tag}] {msg}")


# ---------------------------------------------------------------------------
# Input sanitization helpers
# ---------------------------------------------------------------------------

def safe_choice(value, choices, default):
    return value if value in choices else default


def safe_int(value, default, min_value=None, max_value=None):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if min_value is not None:
        number = max(min_value, number)
    if max_value is not None:
        number = min(max_value, number)
    return number


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_NO_PROXY = {"http": None, "https": None}


def _timeout(read_seconds):
    try:
        read_to = int(read_seconds)
    except (TypeError, ValueError):
        read_to = DEFAULT_READ_TIMEOUT
    if read_to <= 0:
        read_to = DEFAULT_READ_TIMEOUT
    min_connect = min(read_to, 10)
    max_connect = min(read_to, 30)
    connect_to = int(max(min_connect, min(max_connect, read_to // 3)))
    return (connect_to, read_to)


def _request_with_proxy_fallback(method, url, timeout_seconds, **kwargs):
    timeout = _timeout(timeout_seconds)
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        return resp
    except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as e:
        _log("warn", f"Proxy/connection failed, retrying without proxy: {type(e).__name__}")
    kwargs_no_proxy = {k: v for k, v in kwargs.items() if k != "proxies"}
    kwargs_no_proxy["proxies"] = _NO_PROXY
    return requests.request(method, url, timeout=timeout, **kwargs_no_proxy)


def _get(url, timeout_seconds, **kwargs):
    return _request_with_proxy_fallback("GET", url, timeout_seconds, **kwargs)


def _post(url, timeout_seconds, **kwargs):
    return _request_with_proxy_fallback("POST", url, timeout_seconds, **kwargs)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ProxyError,
)


class _EmptyDataRetryableError(Exception):
    pass


def _on_retryable_error(exc):
    exc_type = type(exc).__name__
    exc_msg = str(exc)[:300]
    _log("warn", f"Retryable exception: {exc_type}: {exc_msg}")


def _jittered_backoff_seconds(attempt):
    base = min(2 ** (attempt - 1), 30)
    jitter = random.uniform(0, base * 0.5)
    return base + jitter


def _jittered_sleep(attempt):
    time.sleep(_jittered_backoff_seconds(attempt))


def _is_retryable_http(status):
    return status in (408, 429) or status >= 500


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

def normalize_url(url):
    base = (url or "").strip().rstrip("/")
    if "://" in base:
        scheme = base.split("://")[0].lower()
        if scheme not in ("https", "http"):
            raise ValueError(
                f"Unsupported protocol '{scheme}://', please use https:// or http://"
            )
    elif not base.startswith("http"):
        base = "https://" + base
    if base.endswith("/v1"):
        base = base[:-3]
    return base


# ---------------------------------------------------------------------------
# Image conversion utilities
# ---------------------------------------------------------------------------

def tensor_to_png_bytes(tensor):
    if tensor is None:
        raise ValueError("Input image is empty")
    single = tensor[0:1] if len(tensor.shape) == 4 else tensor.unsqueeze(0)
    arr = (single[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tensor_to_data_url(tensor):
    return "data:image/png;base64," + base64.b64encode(
        tensor_to_png_bytes(tensor)
    ).decode("utf-8")


def image_bytes_to_tensor(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    tensor = torch.from_numpy(arr).unsqueeze(0).float().mul_(1.0 / 255.0)
    return tensor


def _image_bytes_to_uint8(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    return torch.from_numpy(arr).unsqueeze(0)


def _batch_uint8_to_image(tensors):
    if not tensors:
        return torch.empty(0)
    batch = torch.cat(tensors, dim=0)
    return batch.float().mul_(1.0 / 255.0)


def b64_json_to_tensor(b64_json):
    value = (b64_json or "").strip()
    if not value:
        raise ValueError("b64_json is empty")
    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]
    return image_bytes_to_tensor(base64.b64decode(value))


def pil_to_b64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _blank_image_tensor():
    return torch.zeros(1, 64, 64, 3)


def _auto_downscale(image_tensor, max_total_pixels=4 * 1024 * 1024):
    was_3d = image_tensor.dim() == 3
    if was_3d:
        image_tensor = image_tensor.unsqueeze(0)
    samples = image_tensor.movedim(-1, 1)
    h, w = samples.shape[2], samples.shape[3]
    current_pixels = h * w
    if current_pixels <= max_total_pixels:
        return image_tensor.squeeze(0) if was_3d else image_tensor
    scale = math.sqrt(max_total_pixels / current_pixels)
    new_w = round(w * scale)
    new_h = round(h * scale)
    scaled = torch.nn.functional.interpolate(
        samples, size=(new_h, new_w),
        mode="bilinear", align_corners=False,
    )
    return scaled.movedim(1, -1)


def _preprocess_compress_image(image_tensor, target_max_pixels=2 * 1024 * 1024,
                               target_max_bytes=500 * 1024):
    tensor = _auto_downscale(image_tensor, target_max_pixels)
    png_bytes = tensor_to_png_bytes(tensor)
    while len(png_bytes) > target_max_bytes:
        if tensor.dim() == 4:
            h, w = tensor.shape[1], tensor.shape[2]
        else:
            h, w = tensor.shape[0], tensor.shape[1]
        if h <= 64 or w <= 64:
            break
        tensor = _auto_downscale(tensor, max_total_pixels=(h * w) // 4)
        png_bytes = tensor_to_png_bytes(tensor)
    return tensor


def _download_bytes_with_retry(url, timeout_seconds, retry_times=DEFAULT_RETRY_TIMES):
    headers = {
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    last_error = None
    for attempt in range(1, retry_times + 1):
        try:
            response = _get(url, timeout_seconds, headers=headers)
            response.raise_for_status()
            return response.content
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = str(exc)
            _on_retryable_error(exc)
            if attempt < retry_times:
                _jittered_sleep(attempt)
                continue
            break
        except requests.exceptions.HTTPError as exc:
            if _is_retryable_http(exc.response.status_code):
                last_error = str(exc)
                if attempt < retry_times:
                    _jittered_sleep(attempt)
                    continue
            raise RuntimeError(f"Download failed (url={url[:200]}): {exc}") from exc
        except Exception as exc:
            last_error = str(exc)
            if attempt < retry_times:
                _jittered_sleep(attempt)
                continue
            break
    raise RuntimeError(f"Download failed after {retry_times} attempts (url={url[:200]}): {last_error}")


def download_image_with_retry(url, timeout_seconds=60, retry_times=3):
    content = _download_bytes_with_retry(url, timeout_seconds, retry_times)
    return image_bytes_to_tensor(content)


def _download_images_async(urls, timeout_seconds, retry_times=DEFAULT_RETRY_TIMES):
    if not urls:
        return [], []
    tensors = []
    successful_urls = []
    failed_count = 0
    for idx, url in enumerate(urls):
        try:
            content = _download_bytes_with_retry(url, timeout_seconds, retry_times)
            tensors.append(_image_bytes_to_uint8(content))
            successful_urls.append(url)
        except Exception as exc:
            failed_count += 1
            _log("warn", f"Image {idx+1}/{len(urls)} download failed, skipped: {exc}")
            continue
    if not tensors:
        raise RuntimeError(f"All {len(urls)} image downloads failed")
    if failed_count:
        _log("warn", f"Image downloads: {len(tensors)}/{len(urls)} succeeded, {failed_count} skipped")
    return tensors, successful_urls


# ---------------------------------------------------------------------------
# API error extraction
# ---------------------------------------------------------------------------

def _safe_json_dumps(obj, **kwargs):
    kwargs.setdefault("ensure_ascii", False)
    try:
        return json.dumps(obj, **kwargs)
    except (TypeError, ValueError):
        return json.dumps(obj, default=str, **kwargs)


def _sanitize_api_response(data):
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if key == "b64_json":
                continue
            sanitized[key] = _sanitize_api_response(value)
        return sanitized
    if isinstance(data, list):
        return [_sanitize_api_response(item) for item in data]
    return data


def _strip_image_data(obj, max_preview=60):
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k == "b64_json":
                continue
            if k in ("image_data", "init_images", "image_url"):
                if isinstance(v, str) and len(v) > max_preview:
                    cleaned[k] = v[:max_preview] + f"...<truncated {len(v) - max_preview} chars>"
                elif isinstance(v, list):
                    cleaned[k] = [
                        (item[:max_preview] + f"...<truncated {len(item) - max_preview} chars>")
                        if isinstance(item, str) and len(item) > max_preview else item
                        for item in v
                    ]
                else:
                    cleaned[k] = v
            else:
                cleaned[k] = _strip_image_data(v, max_preview)
        return cleaned
    if isinstance(obj, list):
        return [_strip_image_data(item, max_preview) for item in obj]
    return obj


def _extract_api_error_message(data):
    if not isinstance(data, dict):
        return str(data)[:500]
    error = data.get("error")
    if isinstance(error, dict):
        return error.get("message") or _safe_json_dumps(error)
    if isinstance(error, str):
        return error
    return _safe_json_dumps(_sanitize_api_response(data))[:500]


def _safe_extract_error_from_response(response, max_length=500):
    try:
        data = response.json()
        return _extract_api_error_message(data)[:max_length]
    except Exception:
        _log("debug", f"Failed to parse API error response: {type(Exception).__name__}")
        return f"HTTP {response.status_code}"


# ---------------------------------------------------------------------------
# Frontend status emitter
# ---------------------------------------------------------------------------

def emit_runtime_status(node_id, status, message="", elapsed_seconds=0.0,
                        attempt=0, retry_times=0, timeout_seconds=0):
    if node_id in (None, ""):
        return
    try:
        from server import PromptServer
        if PromptServer.instance is None:
            return
        PromptServer.instance.send_sync(
            "comfyui_zhangyuapi_status",
            {
                "node_id": str(node_id),
                "status": status,
                "message": message,
                "elapsed_seconds": float(elapsed_seconds),
                "attempt": int(attempt),
                "retry_times": int(retry_times),
                "timeout_seconds": int(timeout_seconds),
                "timestamp": time.time(),
            },
        )
    except Exception:
        _log("debug", "Failed to send status update (ignorable)")


def _skip_error_return(error_msg, return_types, unique_id=None,
                       retry_times=3, timeout_seconds=360):
    if unique_id:
        emit_runtime_status(unique_id, "error", error_msg,
                            0, 0, retry_times, timeout_seconds)
    blank_img = _blank_image_tensor()
    parts = []
    for t in return_types:
        if t == "IMAGE":
            parts.append(blank_img)
        else:
            parts.append(f"skip_error: {error_msg}")
    return tuple(parts)


# ---------------------------------------------------------------------------
# Response parsing & image download
# ---------------------------------------------------------------------------

def _parse_response_images(data, timeout_seconds, error_prefix="API",
                           unique_id=None, n_expected=None):
    items = data.get("data")
    if not items:
        raise _EmptyDataRetryableError(f"API returned empty data (possible server overload): {data}")
    if not isinstance(items, list):
        items = [items]

    tensors = []
    b64_items = []
    url_items = []
    b64_failed = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            b64_items.append(item["b64_json"])
        elif item.get("url"):
            url_items.append(item["url"])

    for idx, b64 in enumerate(b64_items):
        try:
            tensors.append(_image_bytes_to_uint8(base64.b64decode(
                b64.split(",", 1)[1] if "," in b64 and b64.lower().startswith("data:") else b64
            )))
        except Exception as exc:
            b64_failed += 1
            _log("warn", f"Image {idx+1}/{len(b64_items)} base64 decode failed, skipped: {exc}")

    successful_urls = []
    if url_items:
        if unique_id:
            emit_runtime_status(unique_id, "running",
                                f"Downloading images ({len(url_items)})...", 0, 0, 0, 0)
        url_tensors, successful_urls = _download_images_async(url_items, timeout_seconds)
        tensors.extend(url_tensors)

    if not tensors:
        raise RuntimeError(
            f"Failed to parse any images from {error_prefix} response: "
            f"{_safe_json_dumps(_sanitize_api_response(data))[:500]}"
        )

    failed = len(items) - len(tensors)
    if n_expected is not None and len(tensors) < n_expected:
        failed = n_expected - len(tensors)
    return _batch_uint8_to_image(tensors), successful_urls, failed


# ===================================================================
# Node class
# ===================================================================

class Comfly_gpt_image_2:

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "response", "image_urls", "chats")
    FUNCTION = "generate"
    CATEGORY = "Zaiduyu/Openai"

    IMAGE_SIZES = ["auto (不传size)", "1K", "2K", "4K", "custom_WxH (自定义尺寸)"]
    ASPECT_RATIOS = [
        "1:1",
        "2:3", "3:2", "3:4", "4:5",
        "9:16", "16:9", "21:9",
    ]

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Warning: if key is leaked, regenerate immediately!"
                    }),
                "prompt (提示词)": ("STRING", {"default": "", "multiline": True}),
                "model (模型)": ("STRING", {"default": "gpt-image-2", "multiline": False}),
            },
            "optional": {
                **{f"image{i}": ("IMAGE",) for i in range(1, 9)},
                "mask": ("MASK",),
                "image_size (分辨率)": (
                    cls.IMAGE_SIZES, {"default": "auto (不传size)"}),
                "custom_size (自定义尺寸)": (
                    "STRING", {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Select custom_WxH above, then enter size here, e.g. 1600x1200. Width/height must be multiples of 16, max 3840.",
                    }),
                "aspect_ratio (宽高比)": (
                    cls.ASPECT_RATIOS, {"default": "1:1"}),
                "quality (画质)": (
                    ["auto", "low", "medium", "high"], {"default": "auto"}),
                "response_format (响应格式)": (
                    ["b64_json", "url"], {"default": "b64_json"}),
                "output_format (输出格式)": (
                    ["png", "jpeg", "webp"], {"default": "jpeg"}),
                "output_compression (压缩率)": (
                    "INT", {"default": 85, "min": 0, "max": 100}),
                "moderation (审核模式)": (
                    ["auto", "low"], {"default": "auto"}),
                "seed (本地种子)": (
                    "INT", {
                        "default": 0, "min": 0, "max": 2147483647,
                        "control_after_generate": True,
                    }),
                "timeout_seconds (超时秒数)": (
                    "INT", {
                        "default": DEFAULT_NODE_TIMEOUT,
                        "min": DEFAULT_MIN_NODE_TIMEOUT,
                        "max": DEFAULT_MAX_NODE_TIMEOUT,
                    }),
                "retry_times (重试次数)": (
                    "INT", {"default": DEFAULT_RETRY_TIMES, "min": 1, "max": 10}),
                "skip_error (跳过错误)": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # ------------------------------------------------------------------
    # Size helpers
    # ------------------------------------------------------------------

    SIZE_TABLE = {
        "1K": {
            "AUTO": "auto", "1:1": "1024x1024",
            "2:3": "768x1152", "3:2": "1152x768",
            "3:4": "768x1024", "4:5": "768x960",
            "9:16": "720x1280", "16:9": "1280x720",
            "21:9": "1344x576",
        },
        "2K": {
            "AUTO": "auto", "1:1": "2048x2048",
            "2:3": "1440x2160", "3:2": "2160x1440",
            "3:4": "1536x2048", "4:5": "1536x1920",
            "9:16": "1152x2048", "16:9": "2048x1152",
            "21:9": "2464x1056",
        },
        "4K": {
            "AUTO": "auto", "1:1": "2880x2880",
            "2:3": "2304x3456", "3:2": "3456x2304",
            "3:4": "2448x3264", "4:5": "2304x2880",
            "9:16": "2160x3840", "16:9": "3840x2160",
            "21:9": "3808x1632",
        },
    }

    @staticmethod
    def _validate_custom_size(size_value):
        if not size_value:
            raise ValueError("custom_size is empty")
        size_value = size_value.strip().replace("×", "x").lower()
        import re
        if not re.fullmatch(r"\d{3,4}x\d{3,4}", size_value):
            raise ValueError("custom_size must be like 1600x1200 (width x height)")
        width, height = [int(v) for v in size_value.split("x")]
        max_side = max(width, height)
        min_side = min(width, height)
        total_pixels = width * height
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError("width and height must be multiples of 16")
        if max_side > 3840:
            raise ValueError("max side must not exceed 3840px")
        if max_side / min_side > 3:
            raise ValueError("aspect ratio must not exceed 3:1")
        if total_pixels < 655360 or total_pixels > 8294400:
            raise ValueError("total pixels must be between 655,360 and 8,294,400")
        return f"{width}x{height}"

    def _resolve_size(self, image_size, aspect_ratio, custom_size=""):
        option = (image_size or "").strip().lower()
        ratio = aspect_ratio if aspect_ratio else "1:1"
        if option.startswith("auto"):
            return "auto"
        if "custom" in option:
            return self._validate_custom_size(custom_size)
        tier = None
        if "1k" in option:
            tier = "1K"
        elif "2k" in option:
            tier = "2K"
        elif "4k" in option:
            tier = "4K"
        if tier and ratio in self.SIZE_TABLE[tier]:
            return self.SIZE_TABLE[tier][ratio]
        return "auto"

    # ------------------------------------------------------------------
    # Collect images
    # ------------------------------------------------------------------

    def _collect_images(self, kwargs):
        image_payloads = []
        base64_urls = []
        for i in range(1, 9):
            tensor = kwargs.get(f"image{i}")
            if tensor is None:
                continue
            tensor = _preprocess_compress_image(tensor)
            png_bytes = tensor_to_png_bytes(tensor)
            image_payloads.append((f"image_{i:02d}.png", png_bytes))
            base64_urls.append(
                "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")
            )
        return image_payloads, base64_urls

    def _collect_images_legacy(self, image1, image2, image3, image4,
                               image5=None, image6=None, image7=None, image8=None):
        init_images = []
        image_payloads = []
        for img in [image1, image2, image3, image4, image5, image6, image7, image8]:
            if img is not None:
                tensor = _preprocess_compress_image(img)
                png_bytes = tensor_to_png_bytes(tensor)
                data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")
                init_images.append(data_url)
                image_payloads.append(("image.png", png_bytes))
        return init_images, image_payloads

    # ------------------------------------------------------------------
    # Build payload
    # ------------------------------------------------------------------

    def _build_payload(self, model, prompt, size, quality, response_format,
                       output_format, output_compression, n_images,
                       init_images=None, background="auto", moderation="auto"):
        fields = {"model": model, "prompt": prompt, "n": n_images}
        if size and size != "auto":
            fields["size"] = size
        if quality and quality != "auto":
            fields["quality"] = quality
        # if response_format:
        #     fields["response_format"] = response_format
        fields["output_format"] = output_format
        if output_format == "png":
            fields["output_compression"] = 100
        else:
            fields["output_compression"] = output_compression
        if init_images:
            fields["init_images"] = init_images
        if background and background != "auto":
            fields["background"] = background
        if moderation and moderation != "auto":
            fields["moderation"] = moderation
        fields.pop("referenced_image_ids", None)
        return fields

    # ------------------------------------------------------------------
    # API requests
    # ------------------------------------------------------------------

    def _request_text2img(self, api_base, headers, fields, timeout_seconds):
        url = f"{api_base}/v1/images/generations"
        return _post(url, timeout_seconds, headers={**headers, "Content-Type": "application/json"}, json=fields)

    def _request_img2img(self, api_base, headers, fields, image_payloads,
                         mask_bytes, timeout_seconds):
        url = f"{api_base}/v1/images/edits"
        files = [
            ("image[]", (filename, BytesIO(image_bytes), "image/png"))
            for filename, image_bytes in image_payloads
        ]
        if mask_bytes is not None:
            files.append(("mask", ("mask.png", BytesIO(mask_bytes), "image/png")))
        skip_fields = {"init_images", "referenced_image_ids"}
        data = {}
        for key, value in fields.items():
            if key in skip_fields:
                continue
            if value is None:
                continue
            if isinstance(value, list):
                data[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (int, float, bool)):
                data[key] = value
            else:
                data[key] = str(value)
        return _post(url, timeout_seconds, headers=headers, data=data, files=files)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def generate(self, **kwargs):
        skip_error = kwargs.get("skip_error (跳过错误)", False)
        try:
            return self._generate_impl(**kwargs)
        except Exception as exc:
            if not skip_error:
                raise
            _log("warn", f"skip_error mode, node failed: {exc}")
            return _skip_error_return(
                f"{type(exc).__name__}: {exc}", self.RETURN_TYPES,
                unique_id=kwargs.get("unique_id"),
                retry_times=kwargs.get("retry_times (重试次数)", DEFAULT_RETRY_TIMES),
                timeout_seconds=kwargs.get("timeout_seconds (超时秒数)", DEFAULT_NODE_TIMEOUT),
            )

    def _generate_impl(self, **kwargs):
        pbar = ProgressBar(100) if ProgressBar else None
        unique_id = kwargs.get("unique_id")
        start_ts = time.time()

        api_key = kwargs.get("api_key (API密钥)", "").strip()
        prompt = kwargs.get("prompt (提示词)", "")
        model = (kwargs.get("model (模型)") or "").strip() or "gpt-image-2"
        api_base = normalize_url(DEFAULT_API_BASE_URL)
        image_size = kwargs.get("image_size (分辨率)",
                                kwargs.get("size (分辨率)", "auto (不传size)"))
        aspect_ratio = kwargs.get("aspect_ratio (宽高比)", "1:1")
        quality = safe_choice(
            kwargs.get("quality (画质)", "auto"),
            ["auto", "low", "medium", "high"], "auto")
        response_format = safe_choice(
            kwargs.get("response_format (响应格式)", "b64_json"),
            ["b64_json", "url"], "b64_json")
        output_format = safe_choice(
            kwargs.get("output_format (输出格式)", "jpeg"),
            ["png", "jpeg", "webp"], "jpeg")
        output_compression = safe_int(
            kwargs.get("output_compression (压缩率)", 85), 85, 0, 100)
        moderation = safe_choice(
            kwargs.get("moderation (审核模式)", "auto"),
            ["auto", "low"], "auto")
        seed = safe_int(
            kwargs.get("seed (本地种子)", 0), 0, 0, 2147483647)
        timeout_seconds = safe_int(
            kwargs.get("timeout_seconds (超时秒数)", DEFAULT_NODE_TIMEOUT),
            DEFAULT_NODE_TIMEOUT, DEFAULT_MIN_NODE_TIMEOUT, DEFAULT_MAX_NODE_TIMEOUT)
        retry_times = safe_int(
            kwargs.get("retry_times (重试次数)", DEFAULT_RETRY_TIMES),
            DEFAULT_RETRY_TIMES, 1, 10)
        custom_size = kwargs.get("custom_size (自定义尺寸)", "")

        if not api_key:
            raise ValueError("API Key cannot be empty")
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        # Collect images
        image_payloads, init_images = self._collect_images(kwargs)
        if not image_payloads:
            image1 = kwargs.get("image1")
            image2 = kwargs.get("image2")
            image3 = kwargs.get("image3")
            image4 = kwargs.get("image4")
            image5 = kwargs.get("image5")
            image6 = kwargs.get("image6")
            image7 = kwargs.get("image7")
            image8 = kwargs.get("image8")
            init_images, image_payloads = self._collect_images_legacy(
                image1, image2, image3, image4, image5, image6, image7, image8)

        actual_mode = "img2img" if init_images else "text2img"
        mask = kwargs.get("mask")
        mask_bytes = None
        if mask is not None:
            if len(mask.shape) == 3:
                mask_np = mask[0].cpu().numpy()
            else:
                mask_np = mask.cpu().numpy()
            alpha = ((1.0 - mask_np) * 255).clip(0, 255).astype(np.uint8)
            h, w = alpha.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = 255
            rgba[:, :, 3] = alpha
            buf = BytesIO()
            Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
            mask_bytes = buf.getvalue()

        # Add timestamp to avoid ComfyUI cache
        clean_prompt = f"{prompt}_{int(time.time() * 1000)}"

        # Append aspect ratio to prompt if not default
        if aspect_ratio and aspect_ratio not in ("1:1",):
            clean_prompt = f"{clean_prompt} --ar {aspect_ratio}"

        try:
            effective_size = self._resolve_size(image_size, aspect_ratio, custom_size)
        except ValueError as exc:
            raise ValueError(f"Size parameter error: {exc}")

        headers = {"Authorization": f"Bearer {api_key}"}
        fields = self._build_payload(
            model, clean_prompt, effective_size, quality, response_format,
            output_format, output_compression, 1,
            init_images=init_images,
            background="auto", moderation=moderation,
        )

        if pbar:
            pbar.update_absolute(10)

        _log("info",
             f"[Comfly_gpt_image_2] mode={actual_mode}, "
             f"model={model}, size={effective_size}, "
             f"quality={quality}, response_format={response_format}")
        _log("info", f"[Comfly_gpt_image_2] api_base={api_base}")
        emit_runtime_status(unique_id, "running", "Generating...",
                            0.0, 0, retry_times, timeout_seconds)

        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                if actual_mode == "img2img":
                    response = self._request_img2img(
                        api_base, headers, fields, image_payloads,
                        mask_bytes, timeout_seconds,
                    )
                else:
                    response = self._request_text2img(
                        api_base, headers, fields, timeout_seconds,
                    )

                if response.status_code != 200:
                    if response.status_code == 401:
                        raise RuntimeError(
                            "API Key invalid (401 Unauthorized), please check your key"
                        )
                    if response.status_code == 403:
                        raise RuntimeError(
                            "API access denied (403 Forbidden), please check account permissions"
                        )

                    try:
                        err_data = response.json()
                        err_msg = _extract_api_error_message(err_data)
                    except Exception:
                        err_msg = _safe_extract_error_from_response(response)
                    last_error = f"API error {response.status_code}: {err_msg}"

                    if _is_retryable_http(response.status_code) and attempt < retry_times:
                        emit_runtime_status(
                            unique_id, "running",
                            f"API returned {response.status_code}, retrying ({attempt}/{retry_times})",
                            time.time() - start_ts, attempt, retry_times, timeout_seconds)
                        _jittered_sleep(attempt)
                        continue
                    raise RuntimeError(last_error)

                data = response.json()
                if pbar:
                    pbar.update_absolute(50)

                image_tensor, image_urls, _failed = _parse_response_images(
                    data, timeout_seconds, error_prefix="gpt-image-2",
                    unique_id=unique_id, n_expected=1)
                if pbar:
                    pbar.update_absolute(90)

                elapsed = time.time() - start_ts
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                response_info = (
                    f"## Generation Result ({timestamp})\n\n"
                    f"- **Model**: {model}\n"
                    f"- **Mode**: {actual_mode}\n"
                    f"- **API**: {api_base}\n"
                    f"- **Size**: {image_size}"
                    + (f" -> {effective_size}" if effective_size != image_size else "") + "\n"
                    f"- **Aspect Ratio**: {aspect_ratio}\n"
                    f"- **Quality**: {quality}\n"
                    f"- **Output Format**: {output_format}"
                    + (f" (compression {output_compression})" if output_format != "png" else "") + "\n"
                    + (f"- **Failed**: {_failed}\n" if _failed else "")
                    + (f"- **Reference Images**: {len(image_payloads)}\n" if image_payloads else "")
                    + (f"- **Mask**: yes\n" if mask_bytes is not None else "")
                    + (f"- **Elapsed**: {elapsed:.1f}s (attempt {attempt}/{retry_times})\n"
                       if elapsed > 1 else "")
                )

                emit_runtime_status(
                    unique_id, "success",
                    f"Generated ({elapsed:.1f}s)" + (f", {_failed} failed" if _failed else ""),
                    elapsed, attempt, retry_times, timeout_seconds)
                if pbar:
                    pbar.update_absolute(100)

                image_urls_string = "\n".join(image_urls) if image_urls else ""
                return (image_tensor, response_info, image_urls_string, "")

            except _EmptyDataRetryableError as exc:
                last_error = str(exc)
                _log("warn", f"Empty data retry ({attempt}/{retry_times}): {last_error}")
                if attempt < retry_times:
                    emit_runtime_status(
                        unique_id, "running",
                        f"Server returned empty data, retrying ({attempt}/{retry_times})",
                        time.time() - start_ts, attempt, retry_times, timeout_seconds)
                    _jittered_sleep(attempt)
                    continue
                break
            except RETRYABLE_EXCEPTIONS as exc:
                last_error = str(exc)
                _on_retryable_error(exc)
                if attempt < retry_times:
                    emit_runtime_status(
                        unique_id, "running",
                        f"Network/proxy/timeout, retrying ({attempt}/{retry_times})",
                        time.time() - start_ts, attempt, retry_times, timeout_seconds)
                    _jittered_sleep(attempt)
                    continue
                break
            except RuntimeError:
                raise
            except Exception as exc:
                last_error = str(exc)
                emit_runtime_status(
                    unique_id, "error", last_error,
                    time.time() - start_ts, attempt, retry_times, timeout_seconds)
                raise

        elapsed = time.time() - start_ts
        emit_runtime_status(
            unique_id, "error", f"Failed after {retry_times} attempts",
            elapsed, retry_times, retry_times, timeout_seconds)
        raise RuntimeError(
            f"Comfly_gpt_image_2 failed after {retry_times} attempts, "
            f"last error: {last_error}"
        )


NODE_CLASS_MAPPINGS = {
    "Comfly_gpt_image_2": Comfly_gpt_image_2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfly_gpt_image_2": "Zhixiapi-image-2",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
