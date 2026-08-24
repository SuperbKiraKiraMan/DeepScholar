# 贡献指南 / Contributing

感谢你的关注。这个仓库优先接受能够复现、可审计并且不会泄露用户数据的改动。

## 开始前

1. Fork 仓库并创建独立分支。
2. 阅读 [README.md](README.md)、[SECURITY.md](SECURITY.md) 和 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
3. 复制 .env.example 为本地 .env；不要把真实密钥、论文全文、数据库、模型或运行产物放入提交。
4. 使用 Python 3.12 或更新版本创建虚拟环境并安装 requirements.txt。

## 开发约定

- 优先保持现有 API、SSE 事件名、稳定 source_id 和结构化 schema 的兼容性。
- 不要让 LLM 绕过工具白名单、调用预算、超时、引用检查或最终状态判断。
- 新增或改写的重要代码注释使用中文，并说明业务原因、安全边界或并发约束。
- 公共配置必须使用通用占位符；本机路径和密钥只能出现在被忽略的 .env。
- 新增 provider 或 MCP 工具时，同时补充离线 fixture、错误路径和安全边界测试。

## 验证

提交前至少运行：

~~~bash
python -m pytest tests -m "not openalex_live and not semantic_scholar_live" -q
~~~

如果改动涉及文档、配置或 Docker，请额外检查 README 中的命令、路径和环境变量是否仍然准确。在线 smoke test 只能在明确配置密钥后手动运行，不能作为默认 CI 依赖。

## Pull Request

PR 描述请说明：

- 解决的问题和设计取舍；
- 影响的 API、配置、数据格式或 SSE 事件；
- 已运行的测试及未运行测试的原因；
- 是否涉及本地数据、外部网络或向后兼容风险。

请保持一个 PR 聚焦于一个主题。维护者可能要求拆分过大的改动，或要求补充测试与文档。

## English summary

Please keep changes reproducible and privacy-safe. Preserve public API and SSE contracts, keep LLM actions behind deterministic policy gates, add offline tests for new behavior, and never commit credentials, paper full text, databases, model weights, or generated runtime artifacts. Run the offline pytest suite before opening a pull request and document any untested external integration.
