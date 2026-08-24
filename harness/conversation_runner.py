"""Harness 多轮会话协议运行器。"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.api.schemas import ResearchRequest
from app.observability.lifecycle import emit_after_run, reset_observer, set_observer
from app.services.session_store import SessionExpiredError, SessionNotFoundError, session_store
from harness.fixtures import FixtureManager
from harness.hooks import HookBus
from harness.models import (
    ConversationScenario,
    ConversationScenarioResult,
    ConversationSuiteResult,
    ConversationTurn,
    ConversationTurnResult,
    ExpectationResult,
)


def _expectation(name: str, expected: Any, actual: Any, passed: bool, reason: str = "") -> ExpectationResult:
    return ExpectationResult(
        name=name,
        expected=expected,
        actual=actual,
        passed=passed,
        reason="" if passed else reason,
    )


def _response_value(response: Any, field: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(field, default)
    return getattr(response, field, default)


def _public_sources(response: Any) -> List[Dict[str, Any]]:
    values = _response_value(response, "sources", []) or []
    return [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in values]


async def _run_research_to_completion(request: ResearchRequest, backend: str) -> Dict[str, Any]:
    """通过新的异步 research API 创建 run，并等待后台任务写回完成结果。

    旧版 research_in_session 是同步执行并直接返回完成后的 run；该接口已在
    移除同步端点的清理中删除。新版 research_async 返回 202 + run_id，结果在
    asyncio.create_task 后台任务里写入 run_store。TestClient 会在请求返回时
    取消 create_task，因此这里直接驱动路由函数，并在同一事件循环中 await
    _task_registry 里注册的后台任务（与真实服务器行为一致）。
    """
    from app.api.routes import _task_registry, research_async
    from app.services.run_store import run_store

    response = await research_async(request, backend=backend)
    run_id = _response_value(response, "run_id", "")
    task = _task_registry.get(run_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=120)
    data = run_store.get(run_id) or {}
    data.setdefault("session_id", request.session_id)
    return data


class ConversationScenarioRunner:
    """通过真实 API 函数运行一个共享 Session 的多轮场景。"""

    def __init__(self, hook_bus: Optional[HookBus] = None):
        self.hooks = hook_bus or HookBus()
        self.fixtures = FixtureManager()

    async def run(self, scenario: ConversationScenario) -> ConversationScenarioResult:
        start_ms = int(time.time() * 1000)
        session_id = ""
        isolation_session_id = ""
        turn_results: List[ConversationTurnResult] = []
        scenario_expectations: List[ExpectationResult] = []
        errors: List[str] = []

        self.fixtures.install(scenario.fixture_profile, "rule")
        token = set_observer(self.hooks)
        try:
            # 关键步骤：每个场景创建独立会话，并创建一个哨兵会话验证无跨会话污染。
            session_id = session_store.create(ttl_minutes=scenario.ttl_minutes).session_id
            if scenario.expect_isolated_session:
                isolation_session_id = session_store.create(ttl_minutes=scenario.ttl_minutes).session_id

            for turn in scenario.turns:
                turn_results.append(await self._run_turn(scenario, turn, session_id))

            final = self._session_snapshot(session_id)
            observed_turn_count = final.turn_count if final else (turn_results[-1].session_turn_count if turn_results else 0)
            observed_paper_count = len(final.recommended_papers) if final else 0
            if scenario.expected_final_turn_count is not None:
                scenario_expectations.append(_expectation(
                    "final_turn_count",
                    scenario.expected_final_turn_count,
                    observed_turn_count,
                    observed_turn_count == scenario.expected_final_turn_count,
                    "最终会话轮数不符合场景契约",
                ))
            scenario_expectations.append(_expectation(
                "min_final_papers",
                f"≥{scenario.min_final_papers}",
                observed_paper_count,
                observed_paper_count >= scenario.min_final_papers,
                "累计论文数量不足",
            ))
            if final is not None:
                final_keys = [session_store._paper_key(item) for item in final.recommended_papers]
                expected_preview_count = min(final.turn_count * 2, 20)
                preview_complete = len(final.recent_messages) == expected_preview_count
                scenario_expectations.append(_expectation(
                    "conversation_preview_complete",
                    expected_preview_count,
                    len(final.recent_messages),
                    preview_complete,
                    "并发或顺序更新丢失了会话预览消息",
                ))
                user_message_count = sum(
                    1 for message in final.conversation_messages
                    if message.get("role") == "user"
                )
                transcript_complete = user_message_count >= final.turn_count
                scenario_expectations.append(_expectation(
                    "conversation_transcript_complete",
                    f"≥{final.turn_count}",
                    user_message_count,
                    transcript_complete,
                    "并发或顺序更新丢失了完整会话中的用户消息",
                ))
                if scenario.expect_unique_papers:
                    scenario_expectations.append(_expectation(
                        "unique_accumulated_papers",
                        True,
                        len(final_keys) == len(set(final_keys)),
                        len(final_keys) == len(set(final_keys)),
                        "会话累计论文存在稳定标识重复",
                    ))
            else:
                final_keys = []

            if isolation_session_id:
                isolated = self._session_snapshot(isolation_session_id)
                isolated_clean = bool(
                    isolated is not None
                    and isolated.turn_count == 0
                    and not isolated.recommended_papers
                    and not isolated.active_report_id
                )
                scenario_expectations.append(_expectation(
                    "session_isolation", True, isolated_clean, isolated_clean,
                    "哨兵 Session 被当前场景污染",
                ))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
            final_keys = []
        finally:
            reset_observer(token)
            self.fixtures.restore()
            if session_id:
                session_store.delete(session_id)
            if isolation_session_id:
                session_store.delete(isolation_session_id)

        # Session 已在 finally 清理，因此最终计数从轮次结果中恢复，避免留下测试数据。
        final_turn_count = turn_results[-1].session_turn_count if turn_results else 0
        final_paper_count = turn_results[-1].session_paper_count if turn_results else 0
        hook_summary = dict(Counter(record.stage for record in self.hooks.records))
        passed = (
            not errors
            and all(item.passed for item in turn_results)
            and all(item.passed for item in scenario_expectations)
        )
        return ConversationScenarioResult(
            scenario_id=scenario.id,
            description=scenario.description,
            passed=passed,
            session_id=session_id,
            total_latency_ms=int(time.time() * 1000) - start_ms,
            turns=turn_results,
            final_turn_count=final_turn_count,
            final_paper_count=final_paper_count,
            final_paper_keys=final_keys,
            expectations=scenario_expectations,
            hook_summary=hook_summary,
            error="; ".join(errors),
        )

    async def _run_turn(
        self,
        scenario: ConversationScenario,
        turn: ConversationTurn,
        session_id: str,
    ) -> ConversationTurnResult:
        before = self._session_snapshot(session_id)
        before_keys = {
            session_store._paper_key(item) for item in (before.recommended_papers if before else [])
        }
        before_ids = {
            str(item.get("source_id") or item.get("paper_id") or "")
            for item in (before.recommended_papers if before else [])
            if item.get("source_id") or item.get("paper_id")
        }
        if turn.expire_session_before:
            # 关键步骤：直接推进 TTL 到过去，验证 API 是否返回 410 协议错误。
            session_store._sessions[session_id].expires_at_ms = int(time.time() * 1000) - 1

        hook_start = len(self.hooks.records)
        executions = await asyncio.gather(*[
            self._execute_request(scenario, turn, session_id)
            for _ in range(turn.parallel_requests)
        ])
        after = self._session_snapshot(session_id)
        after_keys = {
            session_store._paper_key(item) for item in (after.recommended_papers if after else [])
        }
        new_paper_count = len(after_keys - before_keys)
        turn_hooks = list(self.hooks.records[hook_start:])
        expectations = self._assert_turn(
            turn, session_id, executions, before_ids, new_paper_count, turn_hooks,
        )

        responses = [item[1] for item in executions if item[1] is not None]
        return ConversationTurnResult(
            turn_id=turn.id,
            passed=all(item.passed for item in expectations),
            http_statuses=[item[0] for item in executions],
            intents=[str(_response_value(item, "intent", "")) for item in responses],
            execution_routes=[str(_response_value(item, "execution_route", "")) for item in responses],
            run_ids=[str(_response_value(item, "run_id", "")) for item in responses],
            session_ids=[str(_response_value(item, "session_id", "")) for item in responses],
            seed_paper_ids=[list(_response_value(item, "seed_paper_ids", []) or []) for item in responses],
            source_counts=[len(_public_sources(item)) for item in responses],
            new_paper_count=new_paper_count,
            session_turn_count=after.turn_count if after else (before.turn_count if before else 0),
            session_paper_count=len(after.recommended_papers) if after else 0,
            expectations=expectations,
            hooks=turn_hooks,
        )

    async def _execute_request(
        self,
        scenario: ConversationScenario,
        turn: ConversationTurn,
        session_id: str,
    ) -> Tuple[int, Any, str]:
        request = ResearchRequest(**turn.request.model_dump(), session_id=session_id)
        try:
            response = await _run_research_to_completion(request, scenario.backend)
            emit_after_run({
                "scenario_id": scenario.id,
                "turn_id": turn.id,
                "run_id": _response_value(response, "run_id", ""),
                "status": _response_value(response, "status", ""),
                "session_id": session_id,
            })
            return 200, response, ""
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            return exc.status_code, None, str(detail.get("error_code") or exc.detail)

    def _assert_turn(
        self,
        turn: ConversationTurn,
        session_id: str,
        executions: List[Tuple[int, Any, str]],
        before_ids: set[str],
        new_paper_count: int,
        hooks: List[Any],
    ) -> List[ExpectationResult]:
        results: List[ExpectationResult] = []
        statuses = [item[0] for item in executions]
        responses = [item[1] for item in executions if item[1] is not None]
        results.append(_expectation(
            "http_status", [turn.expected_http_status] * turn.parallel_requests, statuses,
            all(status == turn.expected_http_status for status in statuses),
            "HTTP 状态码不符合协议",
        ))
        if turn.expected_error_code:
            codes = [item[2] for item in executions]
            results.append(_expectation(
                "error_code", turn.expected_error_code, codes,
                all(code == turn.expected_error_code for code in codes),
                "错误码不符合协议",
            ))
        if responses:
            self._assert_response_fields(results, turn, session_id, responses, before_ids)

        max_new = turn.max_new_papers if turn.max_new_papers is not None else float("inf")
        in_range = turn.min_new_papers <= new_paper_count <= max_new
        results.append(_expectation(
            "new_paper_count",
            f"{turn.min_new_papers}..{turn.max_new_papers if turn.max_new_papers is not None else '∞'}",
            new_paper_count,
            in_range,
            "本轮新增论文数量超出契约范围",
        ))
        if turn.expect_empty_batch:
            source_counts = [len(_public_sources(response)) for response in responses]
            empty = new_paper_count == 0 and all(count == 0 for count in source_counts)
            results.append(_expectation(
                "empty_recommendation_batch", True, empty, empty,
                "空增量推荐仍返回或写入了论文",
            ))
        stages = {record.stage for record in hooks}
        for required in turn.required_hooks:
            results.append(_expectation(
                f"required_hook:{required}", "present",
                "present" if required in stages else "missing",
                required in stages,
                f"本轮缺少 {required} hook",
            ))
        return results

    @staticmethod
    def _assert_response_fields(
        results: List[ExpectationResult],
        turn: ConversationTurn,
        session_id: str,
        responses: List[Any],
        before_ids: set[str],
    ) -> None:
        response_session_ids = [_response_value(item, "session_id", "") for item in responses]
        results.append(_expectation(
            "session_id_stable", session_id, response_session_ids,
            all(value == session_id for value in response_session_ids),
            "响应没有保持场景 Session ID",
        ))
        for field, expected in (
            ("intent", turn.expected_intent),
            ("execution_route", turn.expected_execution_route),
        ):
            if expected:
                actual = [_response_value(item, field, "") for item in responses]
                results.append(_expectation(
                    field, expected, actual, all(value == expected for value in actual),
                    f"{field} 不符合协议",
                ))
        if turn.expected_statuses:
            actual = [_response_value(item, "status", "") for item in responses]
            results.append(_expectation(
                "run_status", turn.expected_statuses, actual,
                all(value in turn.expected_statuses for value in actual),
                "运行状态不在允许集合中",
            ))
        if turn.expected_follow_up is not None:
            actual = [bool(_response_value(item, "is_follow_up", False)) for item in responses]
            results.append(_expectation(
                "is_follow_up", turn.expected_follow_up, actual,
                all(value is turn.expected_follow_up for value in actual),
                "续接标记不符合协议",
            ))
        if turn.expect_answer is not None:
            actual = [bool(str(_response_value(item, "answer", "")).strip()) for item in responses]
            results.append(_expectation(
                "answer_present", turn.expect_answer, actual,
                all(value is turn.expect_answer for value in actual),
                "回答是否为空不符合协议",
            ))
        if turn.expect_seed_from_session:
            seed_lists = [set(_response_value(item, "seed_paper_ids", []) or []) for item in responses]
            # 关键步骤：完整调研必须携带本轮开始前已累计的全部会话论文 ID。
            seeded = bool(before_ids) and bool(seed_lists) and all(
                before_ids.issubset(seed_list) for seed_list in seed_lists
            )
            results.append(_expectation(
                "session_papers_seeded", True, seeded, seeded,
                "完整调研没有携带会话论文 Seed",
            ))

    @staticmethod
    def _session_snapshot(session_id: str):
        try:
            return session_store.get(session_id)
        except (SessionNotFoundError, SessionExpiredError):
            return None


class ConversationSuiteRunner:
    """顺序运行多轮场景；Fixture 会修改进程级组件，因此禁止并行场景。"""

    def __init__(self, scenarios: List[ConversationScenario]):
        self.scenarios = [scenario.model_copy(deep=True) for scenario in scenarios]

    def validate(self) -> List[str]:
        errors: List[str] = []
        ids = [scenario.id for scenario in self.scenarios]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        errors.extend(f"Duplicate scenario id: '{item}'" for item in duplicates)
        return errors

    async def run(self) -> ConversationSuiteResult:
        results = []
        for scenario in self.scenarios:
            results.append(await ConversationScenarioRunner().run(scenario))
        passed = sum(1 for result in results if result.passed)
        return ConversationSuiteResult(
            total_scenarios=len(results),
            passed_scenarios=passed,
            failed_scenarios=len(results) - passed,
            results=results,
        )
