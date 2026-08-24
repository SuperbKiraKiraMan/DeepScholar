"""共享 httpx 客户端工厂的回归测试。

背景：httpx 默认 trust_env=True 会继承 macOS 系统代理，被本地 Clash/Mihomo
劫持导致 ConnectError。build_httpx_client 必须默认直连，仅在显式设置
HTTP(S)_PROXY 时才走代理。
"""

from types import SimpleNamespace

import httpx

from app.tools.http_client import build_httpx_client


def _fake_client_factory(monkeypatch, env):
    """拦截 httpx.AsyncClient 构造，返回记录调用参数的替身。"""
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    calls = []

    def fake_async_client(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(aclose=lambda: None)

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    return calls


def test_client_does_not_inherit_system_proxy(monkeypatch):
    """默认必须直连：trust_env=False，且不读取任何系统代理。"""
    calls = _fake_client_factory(monkeypatch, {})
    client = build_httpx_client(timeout=5.0)
    client.aclose()
    assert calls[0]["timeout"] == 5.0
    assert calls[0]["trust_env"] is False
    assert "proxy" not in calls[0]


def test_client_honors_explicit_proxy_env(monkeypatch):
    """显式设置 HTTPS_PROXY 时仍应走代理（兼容真实代理环境）。"""
    calls = _fake_client_factory(monkeypatch, {"HTTPS_PROXY": "http://proxy.example:8080"})
    client = build_httpx_client(timeout=5.0)
    client.aclose()
    assert calls[0]["trust_env"] is False
    assert calls[0]["proxy"] == "http://proxy.example:8080"
