"""
BM25 关键词检索模块
对 enterprise-attack.json 建立 BM25 索引，支持关键词检索
"""

import json
import os
import pickle
import re
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STIX_FILE = os.path.join(DATA_DIR, "enterprise-attack.json")
INDEX_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")


def _is_revoked_or_deprecated(obj: dict) -> bool:
    """检查对象是否已撤销或废弃"""
    return obj.get("revoked", False) or obj.get("x_mitre_deprecated", False)


def _extract_external_id(obj: dict) -> Optional[str]:
    """从对象的 external_references 中提取 MITRE ID"""
    refs = obj.get("external_references", [])
    if refs and refs[0].get("source_name") == "mitre-attack":
        return refs[0].get("external_id")
    return None


def _extract_url(obj: dict) -> Optional[str]:
    """从对象的 external_references 中提取 MITRE URL"""
    refs = obj.get("external_references", [])
    if refs and refs[0].get("source_name") == "mitre-attack":
        return refs[0].get("url")
    return None


def _tokenize(text: str) -> list:
    """
    中英文混合分词
    - 英文按空格和标点分割，转小写
    - 中文用 jieba 分词
    """
    # 英文部分：按非字母数字分割
    text = text.lower()
    tokens = []

    # 分离英文和中文
    parts = re.findall(r'[a-z0-9]+|[一-鿿]+', text)
    for part in parts:
        if re.match(r'[a-z0-9]+', part):
            # 英文直接作为 token
            tokens.append(part)
        else:
            # 中文用 jieba 分词
            tokens.extend(jieba.lcut(part))

    return [t for t in tokens if len(t) > 1]  # 过滤单字符


def rrf_fusion(multi_results: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion (RRF) 多路结果融合

    Args:
        multi_results: 多路搜索结果列表，每路是 search() 的返回格式
        k: RRF 参数，控制排名靠后的结果权重衰减速度

    Returns:
        融合后的结果列表，按 RRF 分数降序，score 字段更新为 RRF 分数
    """
    # attack_id → (最佳结果, RRF 累计分数)
    score_map = {}
    result_map = {}

    for results in multi_results:
        for rank, r in enumerate(results):
            aid = r["attack_id"]
            rrf_score = 1.0 / (k + rank + 1)
            if aid in score_map:
                score_map[aid] += rrf_score
            else:
                score_map[aid] = rrf_score
                result_map[aid] = r

    # 按 RRF 分数降序排列
    sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)

    fused = []
    for aid in sorted_ids:
        r = result_map[aid].copy()
        r["score"] = round(score_map[aid], 4)
        fused.append(r)

    return fused


def _preprocess_objects(data: dict) -> list:
    """
    预处理 STIX 对象，提取用于检索的字段
    返回文档列表，每个文档包含:
    - attack_id: MITRE ID (如 T1059)
    - name: 名称
    - description: 描述（截取前1500字符）
    - type: 实体类型
    - platforms: 平台列表
    - url: MITRE URL
    - search_text: 用于检索的拼接文本（含关联实体名称）
    """
    documents = []

    # STIX type -> doc type 映射
    type_map = {
        "attack-pattern": "technique",
        "x-mitre-tactic": "tactic",
        "intrusion-set": "group",
        "malware": "malware",
        "tool": "tool",
        "course-of-action": "mitigation",
        "campaign": "campaign",
    }

    # 第一遍：构建 STIX ID → (attack_id, name) 查找表
    id_lookup = {}
    for obj in data.get("objects", []):
        if _is_revoked_or_deprecated(obj):
            continue
        obj_type = obj.get("type", "")
        if obj_type not in type_map:
            continue
        attack_id = _extract_external_id(obj)
        if not attack_id:
            continue
        id_lookup[obj["id"]] = (attack_id, obj.get("name", ""))

    # 第二遍：从 STIX 关系中提取关联实体名称
    related_names = {}  # attack_id → set of related entity names
    for obj in data.get("objects", []):
        if obj.get("type") != "relationship":
            continue
        src = id_lookup.get(obj.get("source_ref"))
        tgt = id_lookup.get(obj.get("target_ref"))
        if src and tgt:
            # 给源实体添加目标实体名称
            related_names.setdefault(src[0], set()).add(tgt[1])
            # 给目标实体添加源实体名称
            related_names.setdefault(tgt[0], set()).add(src[1])

    # 第三遍：构建文档
    for obj in data.get("objects", []):
        if _is_revoked_or_deprecated(obj):
            continue

        obj_type = obj.get("type", "")
        if obj_type not in type_map:
            continue

        attack_id = _extract_external_id(obj)
        if not attack_id:
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")[:1500]
        url = _extract_url(obj) or ""
        doc_type = type_map[obj_type]
        platforms = obj.get("x_mitre_platforms", [])
        aliases = obj.get("aliases", []) or obj.get("x_mitre_aliases", [])

        # 拼接检索文本
        search_text = f"{attack_id} {name} {description}"
        if platforms:
            search_text += " " + " ".join(platforms)
        if aliases:
            search_text += " " + " ".join(aliases)

        # 添加关联实体名称（限制数量避免膨胀）
        related = related_names.get(attack_id, set())
        if related:
            search_text += " " + " ".join(list(related)[:30])

        documents.append({
            "attack_id": attack_id,
            "name": name,
            "description": description,
            "type": doc_type,
            "platforms": platforms,
            "url": url,
            "search_text": search_text,
        })

    return documents


class BM25Search:
    """BM25 检索引擎"""

    def __init__(self):
        self.documents: list = []
        self.corpus_tokens: list = []
        self.bm25: Optional[BM25Okapi] = None

    def build_index(self, data_path: str = STIX_FILE) -> None:
        """
        从 enterprise-attack.json 构建 BM25 索引
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        print("正在构建 BM25 索引...")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 预处理文档
        self.documents = _preprocess_objects(data)
        print(f"  文档数量: {len(self.documents)}")

        # 分词
        print("  正在分词...")
        self.corpus_tokens = [_tokenize(doc["search_text"]) for doc in self.documents]

        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print("  ✅ BM25 索引构建完成")

    def save_index(self, index_path: str = INDEX_FILE) -> None:
        """持久化索引到文件"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "corpus_tokens": self.corpus_tokens,
            }, f)
        print(f"  索引已保存: {index_path}")

    def load_index(self, index_path: str = INDEX_FILE) -> bool:
        """从文件加载索引，返回是否成功"""
        if not os.path.exists(index_path):
            return False
        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
            self.documents = data["documents"]
            self.corpus_tokens = data["corpus_tokens"]
            self.bm25 = BM25Okapi(self.corpus_tokens)
            print(f"✅ BM25 索引已加载 ({len(self.documents)} 个文档)")
            return True
        except Exception as e:
            print(f"⚠️ 索引加载失败: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> list:
        """
        BM25 关键词检索

        Args:
            query: 用户查询文本
            top_k: 返回前 N 个结果

        Returns:
            结果列表，每个元素包含:
            - attack_id, name, description, type, platforms, url
            - score: 归一化后的置信度 [0, 1]
            - snippet: 描述片段
        """
        if not self.bm25:
            raise RuntimeError("索引未初始化，请先调用 build_index() 或 load_index()")

        # 分词
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        # 计算 BM25 原始分数
        scores = self.bm25.get_scores(tokenized_query)

        # 归一化分数到 [0, 1]（用于结果排序，不用于置信度判断）
        max_score = max(scores) if max(scores) > 0 else 1.0
        normalized_scores = scores / max_score

        # 获取 top_k 结果（按原始分数降序）
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if normalized_scores[idx] < 0.01:  # 过滤极低分
                continue
            doc = self.documents[idx]
            results.append({
                "attack_id": doc["attack_id"],
                "name": doc["name"],
                "description": doc["description"],
                "type": doc["type"],
                "platforms": doc["platforms"],
                "url": doc["url"],
                "score": round(float(normalized_scores[idx]), 4),
                "snippet": doc["description"][:200] + "..." if len(doc["description"]) > 200 else doc["description"],
            })

        return results

    def get_index_stats(self) -> dict:
        """返回索引统计信息"""
        counts = {}
        for d in self.documents:
            counts[d["type"]] = counts.get(d["type"], 0) + 1
        return {"total_documents": len(self.documents), **counts}


def build_and_save(data_path: str = STIX_FILE, index_path: str = INDEX_FILE):
    """构建索引并保存（供命令行调用）"""
    engine = BM25Search()
    engine.build_index(data_path)
    engine.save_index(index_path)
    stats = engine.get_index_stats()
    print(f"\n索引统计: {stats}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BM25 检索索引管理")
    parser.add_argument("action", choices=["build", "search"],
                        help="build=构建索引, search=测试搜索")
    parser.add_argument("--query", default="", help="搜索关键词（search 模式）")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    args = parser.parse_args()

    if args.action == "build":
        build_and_save()
    elif args.action == "search":
        engine = BM25Search()
        if not engine.load_index():
            print("索引不存在，正在构建...")
            engine.build_index()
            engine.save_index()

        if args.query:
            results = engine.search(args.query, top_k=args.top_k)
            print(f"\n搜索: {args.query}")
            print(f"结果 ({len(results)} 条):\n")
            for r in results:
                print(f"  [{r['score']:.4f}] {r['attack_id']} - {r['name']} ({r['type']})")
                print(f"         {r['snippet'][:100]}...")
                print()
