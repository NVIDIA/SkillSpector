"""
问答引擎（主入口）
串联所有模块，实现完整的问答流程
"""

import os
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bm25_search import BM25Search, STIX_FILE, INDEX_FILE, rrf_fusion
from utils.graph_query import GraphQuery
from utils.query_parser import parse_query
from utils.answer_generator import generate_answer, generate_answer_stream, generate_low_confidence_answer


class QAEngine:
    """
    问答引擎

    流程:
    1. LLM 意图识别 → 提取关键词和查询类型
    2. 判断是否需要追问
    3. BM25 关键词检索
    4. 置信度判断 (> 0.75 → Neo4j 查询)
    5. LLM 整合回答
    """

    def __init__(
        self,
        neo4j_password: str = "",
        bm25_index_path: str = INDEX_FILE,
        stix_data_path: str = STIX_FILE,
    ):
        """
        初始化问答引擎

        Args:
            neo4j_password: Neo4j 密码（优先使用参数，其次环境变量）
            bm25_index_path: BM25 索引文件路径
            stix_data_path: STIX 数据文件路径
        """
        self.history = []  # 对话历史
        self.stix_data_path = stix_data_path

        # 初始化 BM25 检索引擎
        print("正在初始化 BM25 检索引擎...")
        self.bm25 = BM25Search()
        if not self.bm25.load_index(bm25_index_path):
            if os.path.exists(stix_data_path):
                print("索引不存在，正在从原始数据构建...")
                self.bm25.build_index(stix_data_path)
                self.bm25.save_index(bm25_index_path)
            else:
                raise FileNotFoundError(
                    f"BM25 索引文件和原始数据文件均不存在: {bm25_index_path}, {stix_data_path}"
                )

        # 初始化 Neo4j 图查询
        print("正在连接 Neo4j...")
        try:
            self.graph = GraphQuery(password=neo4j_password)
        except Exception as e:
            print(f"⚠️ Neo4j 连接失败: {e}")
            print("图查询功能将不可用，仅使用 BM25 检索")
            self.graph = None

        print("✅ 问答引擎初始化完成\n")

    def _is_self_intro(self, user_input: str) -> bool:
        """判断是否为自我介绍类问题"""
        keywords = ["你是谁", "你是什么", "介绍一下", "自我介绍", "你叫什么", "你是什么系统", "你是什么平台"]
        return any(k in user_input for k in keywords)

    def _self_intro_answer(self) -> str:
        """返回自我介绍"""
        return (
            "我是 **NTAD 智能攻防问答系统**，基于全球最权威的 MITRE ATT&CK 网络威胁知识库构建。\n\n"
            "### 系统能力\n\n"
            "| 维度 | 说明 |\n"
            "|------|------|\n"
            "| 数据源 | MITRE ATT&CK Enterprise（25,842 个 STIX 对象） |\n"
            "| 知识图谱 | Neo4j 图数据库，7 类节点、6 类关系、18,000+ 边 |\n"
            "| 检索引擎 | BM25Okapi 关键词检索 + RRF 多路融合 |\n"
            "| 智能引擎 | 通义千问 qwen-max 意图识别 + 回答生成 |\n"
            "| 查询路由 | 7 种意图动态图查询（detail/uses/mitigates 等） |\n\n"
            "### 我能帮您\n\n"
            "- 查询攻击技术详情（如：T1059 是什么？）\n"
            "- 了解战术阶段下的技术（如：初始访问有哪些技术？）\n"
            "- 获取防御建议（如：怎么防御 PowerShell 攻击？）\n"
            "- 追踪威胁组织活动（如：APT28 用了哪些技术？）\n"
            "- 分析恶意软件和工具（如：Cobalt Strike 利用了什么技术？）\n\n"
            "请输入您的问题开始探索。"
        )

    def ask(self, user_input: str) -> str:
        """
        处理用户提问

        Args:
            user_input: 用户输入的文本

        Returns:
            回答文本
        """
        # 自我介绍快捷处理
        if self._is_self_intro(user_input):
            answer = self._self_intro_answer()
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": answer})
            return answer

        # 1. LLM 意图识别
        print("🔍 正在分析问题...")
        try:
            parsed = parse_query(user_input, self.history)
        except Exception as e:
            print(f"⚠️ 意图识别失败: {e}，使用简单关键词检索")
            parsed = {
                "query_type": "general",
                "query_focus": "general",
                "entities": [],
                "search_keywords": user_input.split(),
                "search_keywords_variants": [],
                "need_clarification": False,
                "confidence": 0.5,
            }

        print(f"   查询类型: {parsed.get('query_type', 'unknown')}")
        print(f"   查询焦点: {parsed.get('query_focus', 'general')}")
        print(f"   提取实体: {parsed.get('entities', [])}")
        print(f"   检索关键词: {parsed.get('search_keywords', [])}")

        # 2. 判断是否需要追问
        if parsed.get("need_clarification", False):
            clarification = parsed.get("clarification_question", "请提供更多信息")
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": clarification})
            return clarification

        # 3. BM25 多路检索 + RRF 融合
        keywords = parsed.get("search_keywords", [])
        if not keywords:
            keywords = user_input.split()

        # 主查询
        query_text = " ".join(keywords)
        all_queries = [query_text]

        # 变体查询
        variants = parsed.get("search_keywords_variants", [])
        for v in variants:
            if v:
                all_queries.append(" ".join(v))

        print(f"🔍 BM25 检索: {query_text}（共 {len(all_queries)} 路查询）")

        # 多路搜索
        multi_results = [self.bm25.search(q, top_k=10) for q in all_queries]

        # RRF 融合
        if len(multi_results) > 1:
            search_results = rrf_fusion(multi_results)
        else:
            search_results = multi_results[0] if multi_results else []

        if search_results:
            print(f"   找到 {len(search_results)} 条结果")
            print(f"   最高置信度: {search_results[0]['score']:.4f}")
        else:
            print("   未找到匹配结果")

        # 4. 置信度判断（使用 LLM 意图识别的 confidence，而非 BM25 分数）
        llm_confidence = parsed.get("confidence", 0.5)
        has_results = bool(search_results)

        if has_results and llm_confidence >= 0.5:
            # 有检索结果且 LLM 置信度足够 → Neo4j 查询
            print(f"✅ BM25 命中 {len(search_results)} 条，LLM 置信度 {llm_confidence:.2f}，执行图查询")

            graph_data = None
            if self.graph:
                # 收集要查询的 attack_id
                attack_ids = [r["attack_id"] for r in search_results[:5]]

                # 如果有特定实体，优先查询
                entities = parsed.get("entities", [])
                entity_ids = [e for e in entities if any(e.startswith(p) for p in ("T", "TA", "G", "S", "M", "C", "DET"))]
                if entity_ids:
                    attack_ids = entity_ids + attack_ids

                attack_ids = list(dict.fromkeys(attack_ids))  # 去重保序

                print(f"🔍 Neo4j 查询: {attack_ids}")
                graph_data = self.graph.query_by_intent(
                    attack_ids,
                    query_focus=parsed.get("query_focus", "general"),
                    entities=parsed.get("entities", []),
                )

            # LLM 整合回答
            print("🤖 正在生成回答...")
            answer = generate_answer(
                user_question=user_input,
                search_results=search_results,
                graph_data=graph_data,
                history=self.history,
            )
        elif has_results:
            # 有结果但 LLM 置信度低 → 返回检索结果 + 提示
            print(f"⚠️ LLM 置信度 {llm_confidence:.2f} < 0.5，返回检索结果")
            answer = generate_low_confidence_answer(user_input, search_results)
        else:
            # 无检索结果
            print("⚠️ BM25 未找到匹配结果")
            answer = generate_low_confidence_answer(user_input, [])

        # 更新对话历史
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})

        # 限制历史长度
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return answer

    def ask_stream(self, user_input: str):
        """
        流式处理用户提问（逐 token 返回）

        Yields:
            每次产出的文本片段
        """
        # 自我介绍快捷处理（模拟逐字流式）
        if self._is_self_intro(user_input):
            answer = self._self_intro_answer()
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": answer})
            # 按行切分，逐行 yield 实现流式效果
            for line in answer.split("\n"):
                yield line + "\n"
            return

        # 1. LLM 意图识别
        print("🔍 正在分析问题...")
        try:
            parsed = parse_query(user_input, self.history)
        except Exception as e:
            print(f"⚠️ 意图识别失败: {e}，使用简单关键词检索")
            parsed = {
                "query_type": "general",
                "query_focus": "general",
                "entities": [],
                "search_keywords": user_input.split(),
                "search_keywords_variants": [],
                "need_clarification": False,
                "confidence": 0.5,
            }

        print(f"   查询类型: {parsed.get('query_type', 'unknown')}")
        print(f"   查询焦点: {parsed.get('query_focus', 'general')}")

        # 2. 判断是否需要追问
        if parsed.get("need_clarification", False):
            clarification = parsed.get("clarification_question", "请提供更多信息")
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": clarification})
            yield clarification
            return

        # 3. BM25 多路检索 + RRF 融合
        keywords = parsed.get("search_keywords", [])
        if not keywords:
            keywords = user_input.split()

        query_text = " ".join(keywords)
        all_queries = [query_text]
        variants = parsed.get("search_keywords_variants", [])
        for v in variants:
            if v:
                all_queries.append(" ".join(v))

        print(f"🔍 BM25 检索: {query_text}（共 {len(all_queries)} 路查询）")
        multi_results = [self.bm25.search(q, top_k=10) for q in all_queries]
        if len(multi_results) > 1:
            search_results = rrf_fusion(multi_results)
        else:
            search_results = multi_results[0] if multi_results else []

        if search_results:
            print(f"   找到 {len(search_results)} 条结果")
        else:
            print("   未找到匹配结果")

        # 4. 置信度判断
        llm_confidence = parsed.get("confidence", 0.5)
        has_results = bool(search_results)

        if has_results and llm_confidence >= 0.5:
            print(f"✅ BM25 命中 {len(search_results)} 条，LLM 置信度 {llm_confidence:.2f}，执行图查询")

            graph_data = None
            if self.graph:
                attack_ids = [r["attack_id"] for r in search_results[:5]]
                entities = parsed.get("entities", [])
                entity_ids = [e for e in entities if any(e.startswith(p) for p in ("T", "TA", "G", "S", "M", "C", "DET"))]
                if entity_ids:
                    attack_ids = entity_ids + attack_ids
                attack_ids = list(dict.fromkeys(attack_ids))

                print(f"🔍 Neo4j 查询: {attack_ids}")
                graph_data = self.graph.query_by_intent(
                    attack_ids,
                    query_focus=parsed.get("query_focus", "general"),
                    entities=parsed.get("entities", []),
                )

            # 流式生成回答
            print("🤖 正在生成回答...")
            full_answer = ""
            for chunk in generate_answer_stream(
                user_question=user_input,
                search_results=search_results,
                graph_data=graph_data,
                history=self.history,
            ):
                full_answer += chunk
                yield chunk

        elif has_results:
            print(f"⚠️ LLM 置信度 {llm_confidence:.2f} < 0.5，返回检索结果")
            full_answer = generate_low_confidence_answer(user_input, search_results)
            yield full_answer
        else:
            print("⚠️ BM25 未找到匹配结果")
            full_answer = generate_low_confidence_answer(user_input, [])
            yield full_answer

        # 更新对话历史
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": full_answer})
        if len(self.history) > 20:
            self.history = self.history[-20:]

    def clear_history(self):
        """清空对话历史"""
        self.history = []
        print("对话历史已清空")

    def get_stats(self) -> dict:
        """获取引擎统计信息"""
        stats = self.bm25.get_index_stats()
        if self.graph:
            try:
                graph_stats = self.graph.get_stats()
                stats.update(graph_stats)
            except Exception:
                pass
        return stats

    def close(self):
        """关闭连接"""
        if self.graph:
            self.graph.close()


def main():
    """命令行交互入口"""
    import argparse

    parser = argparse.ArgumentParser(description="MITRE ATT&CK 知识图谱问答引擎")
    parser.add_argument("--password", default="", help="Neo4j 密码")
    args = parser.parse_args()

    password = args.password or os.getenv("NEO4J_PASSWORD", "")

    engine = None
    try:
        engine = QAEngine(neo4j_password=password)

        stats = engine.get_stats()
        print(f"📊 引擎统计: {stats}\n")
        print("=" * 50)
        print("MITRE ATT&CK 知识图谱问答系统")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空对话历史")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n👤 您的问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break
            if user_input.lower() == "clear":
                engine.clear_history()
                continue

            print()
            answer = engine.ask(user_input)
            print(f"\n🤖 回答:\n{answer}")

    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if engine:
            engine.close()


if __name__ == "__main__":
    main()
