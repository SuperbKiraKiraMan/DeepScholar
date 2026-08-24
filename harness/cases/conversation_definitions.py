"""多轮会话 Harness 场景定义。"""

from harness.models import ConversationScenario, ConversationTurn, HarnessRequest


def _request(topic: str, max_sources: int = 5) -> HarnessRequest:
    return HarnessRequest(
        topic=topic,
        language="zh",
        max_sources=max_sources,
        run_eval=True,
        agent_mode="rule",
    )


RECOMMEND_MORE_RESEARCH = ConversationScenario(
    id="recommend_more_then_research",
    description="推荐 → 去重后再推荐 → 将累计会话论文作为 Seed 进入完整调研",
    fixture_profile="conversation_default",
    turns=[
        ConversationTurn(
            id="recommend",
            request=_request("请推荐3篇关于RAG evaluation的论文", 3),
            expected_intent="paper_recommendation",
            expected_execution_route="direct_tool",
            expected_follow_up=False,
            expect_answer=True,
            min_new_papers=3,
            max_new_papers=3,
            required_hooks=["before_run", "after_run"],
        ),
        ConversationTurn(
            id="recommend_more",
            request=_request("再推荐2篇", 2),
            expected_intent="recommend_more",
            expected_execution_route="direct_tool",
            expected_follow_up=True,
            expect_answer=True,
            min_new_papers=2,
            max_new_papers=2,
            required_hooks=["before_run", "after_run"],
        ),
        ConversationTurn(
            id="research_from_session",
            request=_request("基于这些论文生成深度调研报告", 5),
            expected_intent="research_from_session",
            expected_execution_route="full_research",
            expected_follow_up=True,
            expect_answer=True,
            expect_seed_from_session=True,
            required_hooks=["before_run", "after_plan", "after_run"],
        ),
    ],
    expected_final_turn_count=3,
    min_final_papers=5,
)


REPORT_FOLLOW_UP = ConversationScenario(
    id="report_follow_up",
    description="先生成报告，再使用 Session 中的报告章节进行追问",
    fixture_profile="conversation_default",
    turns=[
        ConversationTurn(
            id="create_report",
            request=_request("调研 RAG evaluation methods，总结方法和局限", 2),
            expected_intent="deep_research",
            expected_execution_route="full_research",
            expect_answer=True,
            min_new_papers=1,
            required_hooks=["before_run", "after_plan", "after_run"],
        ),
        ConversationTurn(
            id="follow_up",
            request=_request("展开当前报告的方法章节", 2),
            expected_intent="report_follow_up",
            expected_execution_route="conversation",
            expected_follow_up=True,
            expect_answer=True,
            required_hooks=["before_run", "after_run"],
        ),
    ],
    expected_final_turn_count=2,
    min_final_papers=1,
)


SESSION_EXPIRED = ConversationScenario(
    id="session_expired",
    description="过期 Session 必须返回 410 和稳定错误码，且不能记录失败轮次",
    turns=[
        ConversationTurn(
            id="expired_request",
            request=_request("请推荐2篇论文", 2),
            expire_session_before=True,
            expected_http_status=410,
            expected_error_code="SESSION_EXPIRED",
            expected_statuses=[],
        ),
    ],
    expected_final_turn_count=0,
)


CONCURRENT_SESSION_UPDATE = ConversationScenario(
    id="concurrent_session_update",
    description="同一 Session 的并发再推荐必须原子追加、累计轮数且不产生重复论文",
    fixture_profile="conversation_default",
    turns=[
        ConversationTurn(
            id="recommend",
            request=_request("请推荐3篇关于RAG evaluation的论文", 3),
            expected_intent="paper_recommendation",
            expected_execution_route="direct_tool",
            min_new_papers=3,
            max_new_papers=3,
        ),
        ConversationTurn(
            id="parallel_recommend_more",
            request=_request("再推荐2篇", 2),
            parallel_requests=2,
            expected_intent="recommend_more",
            expected_execution_route="direct_tool",
            expected_follow_up=True,
            min_new_papers=4,
            max_new_papers=4,
            required_hooks=["before_run", "after_run"],
        ),
    ],
    expected_final_turn_count=3,
    min_final_papers=7,
)


EMPTY_RECOMMENDATION_BATCH = ConversationScenario(
    id="recommend_more_empty_batch",
    description="候选全部命中历史论文时，返回可解释的空增量且不覆盖累计历史",
    fixture_profile="conversation_empty_batch",
    turns=[
        ConversationTurn(
            id="recommend",
            request=_request("请推荐2篇关于RAG evaluation的论文", 2),
            expected_intent="paper_recommendation",
            expected_execution_route="direct_tool",
            min_new_papers=2,
            max_new_papers=2,
        ),
        ConversationTurn(
            id="empty_more",
            request=_request("再推荐2篇", 2),
            expected_intent="recommend_more",
            expected_execution_route="direct_tool",
            expected_follow_up=True,
            expected_statuses=["failed"],
            expect_answer=True,
            expect_empty_batch=True,
            max_new_papers=0,
        ),
    ],
    expected_final_turn_count=2,
    min_final_papers=2,
)


CROSS_PROVIDER_DUPLICATE = ConversationScenario(
    id="cross_provider_duplicate",
    description="同 DOI 的 Semantic Scholar/OpenAlex 记录只累计一次",
    fixture_profile="conversation_cross_provider_duplicate",
    turns=[
        ConversationTurn(
            id="semantic_batch",
            request=_request("请推荐2篇关于RAG evaluation的论文", 2),
            expected_intent="paper_recommendation",
            expected_execution_route="direct_tool",
            min_new_papers=2,
            max_new_papers=2,
        ),
        ConversationTurn(
            id="openalex_batch",
            request=_request("再推荐2篇", 2),
            expected_intent="recommend_more",
            expected_execution_route="direct_tool",
            expected_follow_up=True,
            min_new_papers=1,
            max_new_papers=1,
        ),
    ],
    expected_final_turn_count=2,
    min_final_papers=3,
)


ALL_CONVERSATION_SCENARIOS = [
    RECOMMEND_MORE_RESEARCH,
    REPORT_FOLLOW_UP,
    SESSION_EXPIRED,
    CONCURRENT_SESSION_UPDATE,
    EMPTY_RECOMMENDATION_BATCH,
    CROSS_PROVIDER_DUPLICATE,
]
