"""
tests/test_planner.py

Planner 测试 —— Phase 1B。
"""

from app.agents.planner import Planner, Task, TaskDAG


class TestPlanner:
    """测试 Planner。"""

    def setup_method(self):
        self.planner = Planner()

    def test_plan_returns_task_dag(self):
        """plan() 返回 TaskDAG。"""
        dag = self.planner.plan("RAG evaluation", max_sources=5)

        assert isinstance(dag, TaskDAG)
        assert dag.topic == "RAG evaluation"
        assert len(dag.tasks) == 4

    def test_tasks_have_correct_types(self):
        """Task DAG 包含四种 task_type。"""
        dag = self.planner.plan("RAG evaluation")

        types = {t.task_type for t in dag.tasks}
        assert types == {"search", "read", "analyze", "cite"}

    def test_tasks_have_correct_dependencies(self):
        """Task 依赖关系正确：search → read → analyze → cite。"""
        dag = self.planner.plan("RAG evaluation")

        search = dag.get_task("search")
        read = dag.get_task("read")
        analyze = dag.get_task("analyze")
        cite = dag.get_task("cite")

        assert search.depends_on == []
        assert read.depends_on == ["search"]
        assert analyze.depends_on == ["read"]
        assert cite.depends_on == ["analyze"]

    def test_tasks_have_tool_plans(self):
        """每个 task 有 tool_plan。"""
        dag = self.planner.plan("RAG evaluation")

        for task in dag.tasks:
            assert len(task.tool_plan) >= 1, f"Task {task.task_id} has no tools"

    def test_get_ready_tasks_empty(self):
        """初始状态：只有 search task 是 ready 的（无依赖）。"""
        dag = self.planner.plan("RAG evaluation")

        ready = dag.get_ready_tasks(set())
        assert len(ready) == 1
        assert ready[0].task_id == "search"

    def test_get_ready_tasks_after_search(self):
        """search 完成后，read 变为 ready。"""
        dag = self.planner.plan("RAG evaluation")

        ready = dag.get_ready_tasks({"search"})
        assert len(ready) == 1
        assert ready[0].task_id == "read"

    def test_get_ready_tasks_after_all_but_cite(self):
        """前三步完成后，只有 cite 是 ready。"""
        dag = self.planner.plan("RAG evaluation")

        ready = dag.get_ready_tasks({"search", "read", "analyze"})
        assert len(ready) == 1
        assert ready[0].task_id == "cite"

    def test_get_ready_tasks_all_done(self):
        """所有 task 完成后，没有 ready task。"""
        dag = self.planner.plan("RAG evaluation")

        ready = dag.get_ready_tasks({"search", "read", "analyze", "cite"})
        assert len(ready) == 0

    def test_task_to_dict(self):
        """Task.to_dict 返回字典。"""
        task = Task("test", "search", "desc", depends_on=["dep1"], tool_plan=["tool1"])
        d = task.to_dict()
        assert d["task_id"] == "test"
        assert d["task_type"] == "search"
        assert d["depends_on"] == ["dep1"]
        assert d["tool_plan"] == ["tool1"]

    def test_dag_to_dict(self):
        """TaskDAG.to_dict 返回字典。"""
        dag = self.planner.plan("RAG evaluation")
        d = dag.to_dict()
        assert d["topic"] == "RAG evaluation"
        assert d["task_count"] == 4
        assert len(d["tasks"]) == 4

    def test_plan_with_different_max_sources(self):
        """不同的 max_sources 不影响 DAG 结构。"""
        dag = self.planner.plan("topic", max_sources=10)
        assert len(dag.tasks) == 4
