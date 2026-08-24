"""
app/services/event_broker.py

RunEventBroker —— 内存版 SSE 事件代理。

职责：
- 按 run_id 保存事件历史
- 为 SSE subscriber 提供 asyncio.Queue
- 支持首次连接回放已有事件
- 支持断线重连后回放未消费事件（Last-Event-ID）
- run_finished 后回放全部事件再关闭
- 慢客户端不阻塞发布端

SSE execution pipeline：负责事件回放、订阅和完成通知。
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional


class RunEventBroker:
    """
    内存版 SSE 事件代理。

    每个 run_id 维护：
    - events: 完整事件历史（用于重连回放）
    - subscribers: asyncio.Queue 列表（活跃 SSE 连接）
    - finished: 是否已结束
    - error: 错误信息
    - cancelled: 是否已取消
    """

    def __init__(self):
        self._runs: Dict[str, Dict[str, Any]] = {}

    def init_run(self, run_id: str, topic: str = ""):
        """初始化 run 的事件记录。"""
        self._runs[run_id] = {
            "events": [],
            "subscribers": [],
            "finished": False,
            "cancelled": False,
            "error": None,
            "topic": topic,
            "created_at": time.time(),
        }

    def run_exists(self, run_id: str) -> bool:
        """检查 run 是否存在（含已结束的 run）。"""
        return run_id in self._runs

    def publish(self, run_id: str, event_type: str, data: Dict[str, Any]):
        """
        发布一个事件到所有 subscriber。

        使用 put_nowait 避免慢客户端阻塞发布端。
        """
        if run_id not in self._runs:
            return

        run = self._runs[run_id]
        seq = len(run["events"]) + 1

        event = {
            "id": str(seq),
            "event": event_type,
            "data": json.dumps(data, default=str, ensure_ascii=False),
            "timestamp_ms": int(time.time() * 1000),
        }

        run["events"].append(event)  # 记录事件历史，供首次连接回放。

        dead_queues = []
        for q in run["subscribers"]: # 推送给所有实时订阅者。
            try:
                q.put_nowait(event) # 通过queue 实现异步通信，避免阻塞发布端
            except asyncio.QueueFull:
                dead_queues.append(q)
            except Exception:
                dead_queues.append(q)

        for q in dead_queues:
            try:
                run["subscribers"].remove(q)
            except ValueError:
                pass

    def subscribe(self, run_id: str) -> Optional[asyncio.Queue]:
        """创建新的 subscriber queue。"""
        if run_id not in self._runs:
            return None
        q = asyncio.Queue(maxsize=256) # 通过queue 实现异步通信，避免阻塞发布端
        self._runs[run_id]["subscribers"].append(q)
        return q

    def unsubscribe(self, run_id: str, queue: asyncio.Queue):
        """移除 subscriber queue。"""
        if run_id not in self._runs:
            return
        try:
            self._runs[run_id]["subscribers"].remove(queue)
        except ValueError:
            pass

    def get_unconsumed_events(
        self, run_id: str, last_event_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取未消费的事件。

        - last_event_id=None 或 "" → 返回所有事件（首次连接）
        - last_event_id="0" → 返回所有事件
        - last_event_id="N" → 返回 id > N 的事件（重连回放）
        """
        if run_id not in self._runs:
            return []

        events = self._runs[run_id]["events"]
        if not last_event_id or last_event_id == "0":
            return list(events)

        try:
            last_seq = int(last_event_id)
        except (ValueError, TypeError):
            return list(events)

        return [e for e in events if int(e["id"]) > last_seq]

    def get_all_events(self, run_id: str) -> List[Dict[str, Any]]:
        """获取所有事件。"""
        if run_id not in self._runs:
            return []
        return list(self._runs[run_id]["events"])

    def cancel_run(self, run_id: str):
        """标记 run 为已取消。"""
        if run_id not in self._runs:
            return
        self._runs[run_id]["cancelled"] = True

    def is_cancelled(self, run_id: str) -> bool:
        """检查 run 是否已取消。"""
        if run_id not in self._runs:
            return True
        return self._runs[run_id]["cancelled"]

    def finish_run(self, run_id: str, error: Optional[str] = None):
        """标记 run 结束。"""
        if run_id not in self._runs:
            return
        self._runs[run_id]["finished"] = True
        self._runs[run_id]["error"] = error

    def is_finished(self, run_id: str) -> bool:
        """检查 run 是否已结束。"""
        if run_id not in self._runs:
            return True
        return self._runs[run_id]["finished"]

    def get_error(self, run_id: str) -> Optional[str]:
        """获取 run 的错误信息。"""
        if run_id not in self._runs:
            return None
        return self._runs[run_id].get("error")

    def cleanup(self, run_id: str):
        """清理 run 的所有订阅者。"""
        if run_id not in self._runs:
            return
        run = self._runs[run_id]
        for q in run["subscribers"]:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        run["subscribers"].clear()

    def get_event_count(self, run_id: str) -> int:
        """获取 run 的事件总数。"""
        if run_id not in self._runs:
            return 0
        return len(self._runs[run_id]["events"])


event_broker = RunEventBroker()
