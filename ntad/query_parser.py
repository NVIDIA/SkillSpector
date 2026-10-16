"""
LLM 意图识别模块
使用千问 API 解析用户提问，提取查询意图和关键实体
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 系统提示词
SYSTEM_PROMPT = """你是一个 MITRE ATT&CK 网络安全知识图谱的查询助手。你的任务是解析用户的问题，提取查询意图和关键实体。

## 你需要输出一个 JSON 对象，包含以下字段：

{
    "query_type": "technique | tactic | group | software | mitigation | campaign | general",
    "query_focus": "detail | uses | mitigates | belongs_to | attributed_to | tactics_of | general",
    "entities": ["提取的实体列表，如 T1059、PowerShell、APT28、Linux 等"],
    "search_keywords": ["用于 BM25 检索的关键词列表（主查询）"],
    "search_keywords_variants": [
        ["变体1：从不同角度或使用同义词的关键词"],
        ["变体2：包含 attack_id 或更具体的术语"]
    ],
    "need_clarification": true/false,
    "clarification_question": "如果需要追问，返回追问内容；否则为空字符串",
    "confidence": 0.0~1.0,
    "reasoning": "简要说明你的判断逻辑"
}

## query_type 说明（用户主要关心哪类实体）：
- technique: 查询攻击技术（如 "T1059 是什么"、"PowerShell 攻击"）
- tactic: 查询战术阶段（如 "初始访问有哪些技术"、"横向移动"）
- group: 查询威胁组织（如 "APT28 的信息"、"哪个组织用了这个技术"）
- software: 查询恶意软件/工具（如 "CobaltStrike 是什么"、"用了哪些工具"）
- mitigation: 查询防御/缓解措施（如 "怎么防御钓鱼"、"T1059 的缓解方法"）
- campaign: 查询攻击活动（如 "最近的攻击活动"）
- general: 通用问题（如 "什么是 ATT&CK"）

## query_focus 说明（用户关心什么关系）：
- detail: 了解实体本身详情（如 "T1059 是什么"、"APT28 介绍"）
- uses: 谁在用/用了什么（如 "APT28 用了哪些技术"、"这个软件利用了什么"）
- mitigates: 防御/缓解（如 "怎么防御 PowerShell"、"T1059 的缓解措施"）
- belongs_to: 归属关系（如 "T1059 属于哪个战术"、"这个技术在哪个阶段"）
- attributed_to: 归属溯源（如 "这个活动是哪个组织干的"）
- tactics_of: 战术下的技术列表（如 "初始访问有哪些技术"）
- general: 通用/不明确

## 注意事项：
1. attack_id 格式：技术 T 开头（T1059、T1059.001），战术 TA 开头（TA0001），组织 G 开头（G0007），软件 S 开头（S0154），缓解 M 开头（M1032），活动 C 开头（C0001）
2. 如果用户问题过于模糊（如 "帮我查一下"），设置 need_clarification=true
3. search_keywords 应该是英文关键词，用于 BM25 检索
4. 如果用户用中文提问，也要提取对应的英文关键词
5. search_keywords_variants 是 search_keywords 的变体，用于多路检索提升召回率。生成 1-2 个变体，每个变体从不同角度选词（如同义词、更具体的 attack_id、上位/下位概念）
6. 只输出 JSON，不要输出其他内容

## 示例：

用户: "T1059 是什么技术？"
输出: {"query_type":"technique","query_focus":"detail","entities":["T1059"],"search_keywords":["T1059","command","scripting","interpreter"],"search_keywords_variants":[["command","script","interpreter","execution"],["T1059.001","PowerShell","T1059.004","Unix shell"]],"need_clarification":false,"clarification_question":"","confidence":0.95,"reasoning":"用户询问特定技术 T1059 的详情"}

用户: "初始访问阶段有哪些攻击技术？"
输出: {"query_type":"tactic","query_focus":"tactics_of","entities":["TA0001"],"search_keywords":["initial","access"],"search_keywords_variants":[["TA0001","phishing","drive-by","exploit"],["spearphishing","watering hole","public-facing"]],"need_clarification":false,"clarification_question":"","confidence":0.9,"reasoning":"用户询问初始访问战术下的技术"}

用户: "PowerShell 相关的攻击怎么防御？"
输出: {"query_type":"mitigation","query_focus":"mitigates","entities":["T1059.001"],"search_keywords":["PowerShell","defense","mitigation"],"search_keywords_variants":[["T1059.001","constrained","language mode"],["scripting","execution","prevention"]],"need_clarification":false,"clarification_question":"","confidence":0.85,"reasoning":"用户询问 PowerShell 攻击的防御方法"}

用户: "APT28 用了哪些攻击技术？"
输出: {"query_type":"group","query_focus":"uses","entities":["G0007"],"search_keywords":["APT28","attack","techniques"],"search_keywords_variants":[["G0007","Fancy Bear","Sofacy"],["APT28","malware","tools","credential"]],"need_clarification":false,"clarification_question":"","confidence":0.95,"reasoning":"用户询问 APT28 组织使用的攻击技术"}

用户: "T1059 属于哪个战术阶段？"
输出: {"query_type":"technique","query_focus":"belongs_to","entities":["T1059"],"search_keywords":["T1059","tactic","phase"],"search_keywords_variants":[["command","scripting","interpreter","execution phase"]],"need_clarification":false,"clarification_question":"","confidence":0.95,"reasoning":"用户询问 T1059 归属的战术阶段"}

用户: "帮我查一下"
输出: {"query_type":"general","query_focus":"general","entities":[],"search_keywords":[],"search_keywords_variants":[],"need_clarification":true,"clarification_question":"请问您想查询什么内容？例如：\n1. 特定攻击技术（如 T1059 PowerShell）\n2. 攻击战术阶段（如初始访问、横向移动）\n3. 威胁组织（如 APT28）\n4. 防御方法","confidence":0.1,"reasoning":"用户问题过于模糊，需要追问"}"""


def _get_client() -> OpenAI:
    """获取千问 API 客户端"""
    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("QWEN_MODEL", "qwen-plus")

    if not api_key or api_key == "your_api_key_here":
        raise ValueError("请在 .env 文件中配置 QWEN_API_KEY")

    return OpenAI(api_key=api_key, base_url=base_url), model


def parse_query(user_input: str, history: list = None) -> dict:
    """
    用千问 API 解析用户提问

    Args:
        user_input: 用户输入的文本
        history: 对话历史 [{"role": "user/assistant", "content": "..."}]

    Returns:
        {
            "query_type": "technique" | "tactic" | "group" | "software" | "mitigation" | "campaign" | "general",
            "query_focus": "detail" | "uses" | "mitigates" | "belongs_to" | "attributed_to" | "tactics_of" | "general",
            "entities": ["T1059", "PowerShell", ...],
            "search_keywords": ["command", "scripting", ...],
            "need_clarification": True/False,
            "clarification_question": "...",
            "confidence": 0.0~1.0,
            "reasoning": "..."
        }
    """
    client, model = _get_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 添加对话历史
    if history:
        messages.extend(history[-6:])  # 最近 3 轮对话

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()

        # 提取 JSON（处理可能的 markdown 代码块）
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        # 验证必要字段
        required_fields = ["query_type", "entities", "search_keywords", "need_clarification", "confidence"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"缺少字段: {field}")

        # query_focus 默认值
        if "query_focus" not in result:
            result["query_focus"] = "general"

        # search_keywords_variants 默认值
        if "search_keywords_variants" not in result:
            result["search_keywords_variants"] = []

        # 确保 confidence 在合理范围
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))

        return result

    except json.JSONDecodeError as e:
        return {
            "query_type": "general",
            "query_focus": "general",
            "entities": [],
            "search_keywords": user_input.split(),
            "search_keywords_variants": [],
            "need_clarification": False,
            "clarification_question": "",
            "confidence": 0.3,
            "reasoning": f"LLM 输出解析失败: {e}",
            "raw_output": content if 'content' in dir() else "",
        }
    except Exception as e:
        return {
            "query_type": "general",
            "query_focus": "general",
            "entities": [],
            "search_keywords": user_input.split(),
            "search_keywords_variants": [],
            "need_clarification": False,
            "clarification_question": "",
            "confidence": 0.3,
            "reasoning": f"LLM 调用失败: {e}",
        }


if __name__ == "__main__":
    # 测试意图识别
    test_queries = [
        "T1059 是什么技术？",
        "初始访问阶段有哪些攻击技术？",
        "PowerShell 相关的攻击怎么防御？",
        "帮我查一下",
        "横向移动 Linux 平台",
    ]

    for query in test_queries:
        print(f"\n问题: {query}")
        result = parse_query(query)
        print(f"  类型: {result['query_type']} | 焦点: {result.get('query_focus', 'N/A')}")
        print(f"  实体: {result['entities']}")
        print(f"  关键词: {result['search_keywords']}")
        print(f"  置信度: {result['confidence']}")
        print(f"  需追问: {result['need_clarification']}")
