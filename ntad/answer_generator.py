"""
LLM 回答生成模块
将 BM25 检索结果 + Neo4j 图查询结果整合，用千问 API 生成自然语言回答
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 系统提示词
SYSTEM_PROMPT = """你是 **NTAD 智能攻防问答系统**，一个基于 MITRE ATT&CK 知识图谱的智能安全问答助手。

## 自我介绍（当用户问"你是谁"、"你是什么"、"介绍一下"等问题时）：

我是 NTAD 智能攻防问答系统，基于全球最权威的 MITRE ATT&CK 网络威胁知识库构建。系统将 25,000+ 个安全实体（攻击技术、威胁组织、恶意软件、防御措施等）构建为知识图谱，存储在 Neo4j 图数据库中，支持自然语言提问和智能检索。我能帮您查询攻击技术详情、追踪威胁组织活动、获取防御建议，实现从问题到结构化回答的端到端智能分析。

## 回答规则：

1. **引用来源**：回答中必须引用相关的 attack_id（如 T1059、TA0002）
2. **结构化输出**：使用清晰的层次结构（标题、列表、表格）
3. **专业但易懂**：技术术语要解释，让非安全专业人员也能理解
4. **实用建议**：在解释攻击技术的同时，提供防御建议
5. **承认不确定性**：如果检索结果不够充分，明确告知用户

## 回答格式：

- 使用 Markdown 格式
- 包含"相关技术"或"相关战术"部分，列出 attack_id 和名称
- 如果有防御建议，单独列出
- 末尾附上 MITRE ATT&CK 参考链接

## 注意事项：
- 不要编造不在检索结果中的技术
- 如果置信度较低，在回答开头说明
- 中文回答，专业术语保留英文"""


def _get_client() -> tuple:
    """获取千问 API 客户端"""
    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("QWEN_MODEL", "qwen-plus")

    if not api_key or api_key == "your_api_key_here":
        raise ValueError("请在 .env 文件中配置 QWEN_API_KEY")

    return OpenAI(api_key=api_key, base_url=base_url), model


def _format_search_context(search_results: list, graph_data: dict = None) -> str:
    """
    将检索结果格式化为 LLM 上下文（带去重和 token 预算）

    Args:
        search_results: BM25 检索结果
        graph_data: Neo4j 图查询结果

    Returns:
        格式化的上下文文本
    """
    # 全局 token 预算（约 6000 中文字符）
    max_chars = 6000
    current_chars = 0
    context_parts = []

    def _can_append(text: str) -> bool:
        """检查是否超出预算"""
        nonlocal current_chars
        if current_chars + len(text) > max_chars:
            return False
        current_chars += len(text)
        return True

    # 收集图查询结果中的 attack_id，用于 BM25 去重
    graph_ids = set()
    if graph_data:
        for key in ["techniques", "tactics", "groups", "software", "mitigations", "campaigns"]:
            for item in graph_data.get(key, []):
                graph_ids.add(item["attack_id"])

    # BM25 检索结果（排除已在图结果中的实体）
    if search_results:
        filtered = [r for r in search_results[:5] if r["attack_id"] not in graph_ids]
        if filtered:
            header = "## BM25 检索结果\n"
            if _can_append(header):
                context_parts.append(header)
                for i, r in enumerate(filtered, 1):
                    entry = (
                        f"{i}. **{r['attack_id']}** - {r['name']} (置信度: {r['score']:.2f})\n"
                        f"   类型: {r['type']}\n"
                        f"   描述: {r['snippet']}\n"
                        f"   链接: {r['url']}\n"
                    )
                    if not _can_append(entry):
                        break
                    context_parts.append(entry)

    # Neo4j 图查询结果
    if graph_data:
        # 技术详情
        if graph_data.get("techniques"):
            header = "\n## 技术详情\n"
            if _can_append(header):
                context_parts.append(header)
                for tech in graph_data["techniques"][:15]:
                    entry = (
                        f"### {tech['attack_id']} - {tech['name']}\n"
                        f"- 平台: {', '.join(tech.get('platforms', []))}\n"
                        f"- 描述: {tech.get('description', '')[:200]}\n"
                    )
                    if not _can_append(entry):
                        break
                    context_parts.append(entry)

        # 关联战术
        if graph_data.get("tactics"):
            header = "\n## 关联战术\n"
            if _can_append(header):
                context_parts.append(header)
                for tac in graph_data["tactics"][:10]:
                    entry = f"- **{tac['attack_id']}** - {tac['name']}\n"
                    if not _can_append(entry):
                        break
                    context_parts.append(entry)

        # 关联组织
        if graph_data.get("groups"):
            header = "\n## 关联威胁组织\n"
            if _can_append(header):
                context_parts.append(header)
                for grp in graph_data["groups"][:10]:
                    aliases = ", ".join(grp.get("aliases", [])[:3])
                    entry = (
                        f"- **{grp['attack_id']}** - {grp['name']}"
                        f"{f' (别名: {aliases})' if aliases else ''}\n"
                    )
                    if not _can_append(entry):
                        break
                    context_parts.append(entry)

        # 关联软件
        if graph_data.get("software"):
            header = "\n## 关联软件\n"
            if _can_append(header):
                context_parts.append(header)
                for sw in graph_data["software"][:15]:
                    sw_type = "恶意软件" if sw.get("type") == "malware" else "工具"
                    entry = (
                        f"- **{sw['attack_id']}** - {sw['name']} [{sw_type}]\n"
                        f"  描述: {sw.get('description', '')[:150]}\n"
                    )
                    if not _can_append(entry):
                        break
                    context_parts.append(entry)

        # 缓解措施
        if graph_data.get("mitigations"):
            header = "\n## 缓解措施\n"
            if _can_append(header):
                context_parts.append(header)
                for mit in graph_data["mitigations"][:10]:
                    entry = (
                        f"- **{mit['attack_id']}** - {mit['name']}\n"
                        f"  {mit.get('description', '')[:150]}\n"
                    )
                    if not _can_append(entry):
                        break
                    context_parts.append(entry)

        # 攻击活动
        if graph_data.get("campaigns"):
            header = "\n## 关联攻击活动\n"
            if _can_append(header):
                context_parts.append(header)
                for camp in graph_data["campaigns"][:10]:
                    entry = f"- **{camp['attack_id']}** - {camp['name']}\n"
                    if not _can_append(entry):
                        break
                    context_parts.append(entry)

        # 关系（优先级最低，放在最后）
        if graph_data.get("relationships"):
            header = "\n## 关系\n"
            if _can_append(header):
                context_parts.append(header)
                for rel in graph_data["relationships"][:30]:
                    entry = f"- {rel['from']} → [{rel['type']}] → {rel['to']} ({rel.get('to_name', '')})\n"
                    if not _can_append(entry):
                        break
                    context_parts.append(entry)
                total = len(graph_data["relationships"])
                if total > 30:
                    overflow = f"\n（共 {total} 条关系，仅展示前 30 条）\n"
                    if _can_append(overflow):
                        context_parts.append(overflow)

    return "\n".join(context_parts)


def _build_messages(user_question: str, search_results: list, graph_data: dict = None, history: list = None) -> list:
    """构建 LLM 消息列表"""
    context = _format_search_context(search_results, graph_data)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    user_message = f"""用户问题: {user_question}

以下是检索到的相关信息：

{context}

请基于以上信息回答用户的问题。"""
    messages.append({"role": "user", "content": user_message})
    return messages


def generate_answer(
    user_question: str,
    search_results: list,
    graph_data: dict = None,
    history: list = None,
) -> str:
    """
    整合检索结果，生成最终回答

    Args:
        user_question: 用户原始问题
        search_results: BM25 检索结果
        graph_data: Neo4j 图查询结果
        history: 对话历史

    Returns:
        自然语言回答
    """
    client, model = _get_client()
    messages = _build_messages(user_question, search_results, graph_data, history)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        fallback = f"⚠️ LLM 回答生成失败: {e}\n\n以下是检索结果摘要：\n\n"
        for r in search_results[:3]:
            fallback += f"- **{r['attack_id']}** - {r['name']} (置信度: {r['score']:.2f})\n"
            fallback += f"  {r['snippet'][:100]}...\n\n"
        return fallback


def generate_answer_stream(
    user_question: str,
    search_results: list,
    graph_data: dict = None,
    history: list = None,
):
    """
    流式生成回答（逐 token 返回）

    Yields:
        每次产出的文本片段
    """
    client, model = _get_client()
    messages = _build_messages(user_question, search_results, graph_data, history)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"⚠️ LLM 回答生成失败: {e}\n\n以下是检索结果摘要：\n\n"
        for r in search_results[:3]:
            yield f"- **{r['attack_id']}** - {r['name']} (置信度: {r['score']:.2f})\n"
            yield f"  {r['snippet'][:100]}...\n\n"


def generate_low_confidence_answer(user_question: str, search_results: list) -> str:
    """
    低置信度时的回答

    当 BM25 检索置信度 < 0.7 时，返回检索结果并提示信息不足
    """
    if not search_results:
        return (
            "抱歉，我没有找到与您问题相关的 MITRE ATT&CK 技术信息。\n\n"
            "建议您：\n"
            "1. 尝试使用更具体的关键词（如具体的攻击技术名称或 attack_id）\n"
            "2. 参考 MITRE ATT&CK 官网: https://attack.mitre.org/\n"
            "3. 描述您想了解的攻击场景或技术特点"
        )

    answer = "🔍 检索结果置信度较低，以下是最接近的信息：\n\n"

    for i, r in enumerate(search_results[:5], 1):
        answer += f"**{i}. {r['attack_id']}** - {r['name']} (置信度: {r['score']:.2f})\n"
        answer += f"   {r['snippet'][:150]}...\n\n"

    answer += (
        "\n💡 以上结果可能不完全匹配您的问题。建议：\n"
        "1. 提供更具体的关键词\n"
        "2. 明确您想了解的攻击技术或战术阶段\n"
        "3. 使用 attack_id（如 T1059）进行精确查询"
    )

    return answer
