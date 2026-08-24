"""共享 httpx 客户端工厂。

背景：httpx 默认 trust_env=True 会经 urllib 继承 macOS 系统代理
（SystemConfiguration / scutil）。本机 Clash/Mihomo（如 127.0.0.1:7897）
对部分 httpx 的 HTTP CONNECT 握手会直接断开（httpcore EndOfStream），导致
OpenAlex / Semantic Scholar / LLM / Crossref 等外部 HTTP 请求全部 ConnectError；
而 curl 不走系统代理所以正常。

策略：统一改为不继承系统代理，只在显式设置 HTTP(S)_PROXY 环境变量时才使用代理。
这样在真实网络环境直连、在需要代理的环境显式配置，行为都可预期。
"""

from __future__ import annotations

import os
from typing import Any

import httpx


def build_httpx_client(*, timeout: float, **kwargs: Any) -> httpx.AsyncClient:
    """构造不继承系统代理的 httpx.AsyncClient。

    - trust_env=False：忽略 macOS/系统级代理配置，避免被本地代理误劫持。
    - 仅当显式设置了 HTTPS_PROXY / HTTP_PROXY（含小写变体）时才走代理。
    - 其余参数（headers 等）原样透传给 httpx.AsyncClient。
    """
    proxy = (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )
    if proxy:
        kwargs.setdefault("proxy", proxy)
    return httpx.AsyncClient(timeout=timeout, trust_env=False, **kwargs)
