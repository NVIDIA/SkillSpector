"""
MITRE ATT&CK 知识图谱构建模块
将 Enterprise ATT&CK 数据导入 Neo4j 图数据库
"""

import json
import os
import sys
from typing import Optional

from neo4j import GraphDatabase

# 默认连接配置
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"

# 数据文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STIX_FILE = os.path.join(DATA_DIR, "enterprise-attack.json")


def _extract_external_id(obj: dict) -> Optional[str]:
    """从对象的 external_references 中提取 MITRE ID（如 T1059、TA0001）"""
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


def _is_revoked_or_deprecated(obj: dict) -> bool:
    """检查对象是否已撤销或废弃"""
    return obj.get("revoked", False) or obj.get("x_mitre_deprecated", False)


class GraphBuilder:
    """Neo4j 图数据库构建器"""

    def __init__(self, uri: str = DEFAULT_URI, user: str = DEFAULT_USER, password: str = ""):
        if not password:
            password = os.environ.get("NEO4J_PASSWORD", "")
        if not password:
            raise ValueError("请提供 Neo4j 密码（参数或 NEO4J_PASSWORD 环境变量）")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print(f"✅ 已连接 Neo4j: {uri}")
        except Exception as e:
            raise ConnectionError(f"Neo4j 连接失败: {e}")

    def close(self):
        if hasattr(self, "driver"):
            self.driver.close()

    def _create_constraints(self):
        """为所有节点类型创建唯一约束"""
        constraints = [
            ("Technique", "attack_id"),
            ("Tactic", "attack_id"),
            ("Group", "attack_id"),
            ("Malware", "attack_id"),
            ("Tool", "attack_id"),
            ("Mitigation", "attack_id"),
            ("Campaign", "attack_id"),
        ]
        with self.driver.session() as session:
            for label, prop in constraints:
                query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                session.run(query)
                print(f"  创建约束: {label}.{prop}")

    def _parse_attack_data(self) -> dict:
        """
        解析 MITRE ATT&CK STIX 数据
        返回所有节点类型和关系数据
        """
        if not os.path.exists(STIX_FILE):
            raise FileNotFoundError(f"数据文件不存在: {STIX_FILE}\n请先运行 utils/data_fetcher.py 下载数据")

        print("正在解析 MITRE ATT&CK 数据...")
        with open(STIX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        techniques = []
        tactics = []
        groups = []
        malware_list = []
        tools = []
        mitigations = []
        campaigns = []
        relationships = []

        # stix_id -> attack_id 映射，用于解析 relationship
        stix_to_attack = {}

        for obj in data.get("objects", []):
            obj_type = obj.get("type", "")
            stix_id = obj.get("id", "")

            # relationship 单独收集（不排除 deprecated，但排除 revoked）
            if obj_type == "relationship":
                if not obj.get("revoked", False):
                    relationships.append(obj)
                continue

            if _is_revoked_or_deprecated(obj):
                continue

            attack_id = _extract_external_id(obj)

            if obj_type == "attack-pattern":
                techniques.append(obj)
                if attack_id:
                    stix_to_attack[stix_id] = attack_id
            elif obj_type == "x-mitre-tactic":
                tactics.append(obj)
                if attack_id:
                    stix_to_attack[stix_id] = attack_id
            elif obj_type == "intrusion-set":
                groups.append(obj)
                if attack_id:
                    stix_to_attack[stix_id] = attack_id
            elif obj_type == "malware":
                malware_list.append(obj)
                if attack_id:
                    stix_to_attack[stix_id] = attack_id
            elif obj_type == "tool":
                tools.append(obj)
                if attack_id:
                    stix_to_attack[stix_id] = attack_id
            elif obj_type == "course-of-action":
                mitigations.append(obj)
                if attack_id:
                    stix_to_attack[stix_id] = attack_id
            elif obj_type == "campaign":
                campaigns.append(obj)
                if attack_id:
                    stix_to_attack[stix_id] = attack_id

        print(f"  技术: {len(techniques)}")
        print(f"  战术: {len(tactics)}")
        print(f"  威胁组织: {len(groups)}")
        print(f"  恶意软件: {len(malware_list)}")
        print(f"  工具: {len(tools)}")
        print(f"  缓解措施: {len(mitigations)}")
        print(f"  攻击活动: {len(campaigns)}")
        print(f"  关系: {len(relationships)}")

        return {
            "techniques": techniques,
            "tactics": tactics,
            "groups": groups,
            "malware": malware_list,
            "tools": tools,
            "mitigations": mitigations,
            "campaigns": campaigns,
            "relationships": relationships,
            "stix_to_attack": stix_to_attack,
        }

    def _import_tactics(self, tactics: list) -> int:
        """批量导入战术节点"""
        batch = []
        for t in tactics:
            attack_id = _extract_external_id(t)
            if not attack_id:
                continue
            batch.append({
                "attack_id": attack_id,
                "name": t.get("name", ""),
                "description": t.get("description", "")[:500],
                "url": _extract_url(t) or "",
                "stix_id": t.get("id", ""),
                "shortname": t.get("x_mitre_shortname", ""),
            })

        query = """
        UNWIND $batch AS item
        MERGE (t:Tactic {attack_id: item.attack_id})
        SET t.name = item.name,
            t.description = item.description,
            t.url = item.url,
            t.stix_id = item.stix_id,
            t.shortname = item.shortname
        """

        with self.driver.session() as session:
            session.run(query, batch=batch)

        return len(batch)

    def _import_techniques(self, techniques: list) -> int:
        """批量导入技术节点"""
        batch = []
        for t in techniques:
            attack_id = _extract_external_id(t)
            if not attack_id:
                continue
            batch.append({
                "attack_id": attack_id,
                "name": t.get("name", ""),
                "description": t.get("description", "")[:500],
                "url": _extract_url(t) or "",
                "stix_id": t.get("id", ""),
                "platforms": t.get("x_mitre_platforms", []),
                "is_subtechnique": t.get("x_mitre_is_subtechnique", False),
            })

        query = """
        UNWIND $batch AS item
        MERGE (t:Technique {attack_id: item.attack_id})
        SET t.name = item.name,
            t.description = item.description,
            t.url = item.url,
            t.stix_id = item.stix_id,
            t.platforms = item.platforms,
            t.is_subtechnique = item.is_subtechnique
        """

        with self.driver.session() as session:
            session.run(query, batch=batch)

        return len(batch)

    def _import_groups(self, groups: list) -> int:
        """批量导入威胁组织节点"""
        batch = []
        for g in groups:
            attack_id = _extract_external_id(g)
            if not attack_id:
                continue
            batch.append({
                "attack_id": attack_id,
                "name": g.get("name", ""),
                "description": g.get("description", "")[:500],
                "url": _extract_url(g) or "",
                "stix_id": g.get("id", ""),
                "aliases": g.get("aliases", []),
            })

        query = """
        UNWIND $batch AS item
        MERGE (g:Group {attack_id: item.attack_id})
        SET g.name = item.name,
            g.description = item.description,
            g.url = item.url,
            g.stix_id = item.stix_id,
            g.aliases = item.aliases
        """

        with self.driver.session() as session:
            session.run(query, batch=batch)

        return len(batch)

    def _import_malware(self, malware_list: list) -> int:
        """批量导入恶意软件节点"""
        batch = []
        for m in malware_list:
            attack_id = _extract_external_id(m)
            if not attack_id:
                continue
            batch.append({
                "attack_id": attack_id,
                "name": m.get("name", ""),
                "description": m.get("description", "")[:500],
                "url": _extract_url(m) or "",
                "stix_id": m.get("id", ""),
                "platforms": m.get("x_mitre_platforms", []),
                "aliases": m.get("x_mitre_aliases", []),
            })

        query = """
        UNWIND $batch AS item
        MERGE (m:Malware {attack_id: item.attack_id})
        SET m.name = item.name,
            m.description = item.description,
            m.url = item.url,
            m.stix_id = item.stix_id,
            m.platforms = item.platforms,
            m.aliases = item.aliases
        """

        with self.driver.session() as session:
            session.run(query, batch=batch)

        return len(batch)

    def _import_tools(self, tools: list) -> int:
        """批量导入工具节点"""
        batch = []
        for t in tools:
            attack_id = _extract_external_id(t)
            if not attack_id:
                continue
            batch.append({
                "attack_id": attack_id,
                "name": t.get("name", ""),
                "description": t.get("description", "")[:500],
                "url": _extract_url(t) or "",
                "stix_id": t.get("id", ""),
                "platforms": t.get("x_mitre_platforms", []),
                "aliases": t.get("x_mitre_aliases", []),
            })

        query = """
        UNWIND $batch AS item
        MERGE (t:Tool {attack_id: item.attack_id})
        SET t.name = item.name,
            t.description = item.description,
            t.url = item.url,
            t.stix_id = item.stix_id,
            t.platforms = item.platforms,
            t.aliases = item.aliases
        """

        with self.driver.session() as session:
            session.run(query, batch=batch)

        return len(batch)

    def _import_mitigations(self, mitigations: list) -> int:
        """批量导入缓解措施节点"""
        batch = []
        for m in mitigations:
            attack_id = _extract_external_id(m)
            if not attack_id:
                continue
            batch.append({
                "attack_id": attack_id,
                "name": m.get("name", ""),
                "description": m.get("description", "")[:500],
                "url": _extract_url(m) or "",
                "stix_id": m.get("id", ""),
            })

        query = """
        UNWIND $batch AS item
        MERGE (m:Mitigation {attack_id: item.attack_id})
        SET m.name = item.name,
            m.description = item.description,
            m.url = item.url,
            m.stix_id = item.stix_id
        """

        with self.driver.session() as session:
            session.run(query, batch=batch)

        return len(batch)

    def _import_campaigns(self, campaigns: list) -> int:
        """批量导入攻击活动节点"""
        batch = []
        for c in campaigns:
            attack_id = _extract_external_id(c)
            if not attack_id:
                continue
            batch.append({
                "attack_id": attack_id,
                "name": c.get("name", ""),
                "description": c.get("description", "")[:500],
                "url": _extract_url(c) or "",
                "stix_id": c.get("id", ""),
                "first_seen": c.get("first_seen", ""),
                "last_seen": c.get("last_seen", ""),
                "aliases": c.get("aliases", []),
            })

        query = """
        UNWIND $batch AS item
        MERGE (c:Campaign {attack_id: item.attack_id})
        SET c.name = item.name,
            c.description = item.description,
            c.url = item.url,
            c.stix_id = item.stix_id,
            c.first_seen = item.first_seen,
            c.last_seen = item.last_seen,
            c.aliases = item.aliases
        """

        with self.driver.session() as session:
            session.run(query, batch=batch)

        return len(batch)

    def _import_technique_tactic_relations(self, techniques: list) -> int:
        """导入技术-战术关系：(Technique)-[:BELONGS_TO]->(Tactic)"""
        relations = []
        for t in techniques:
            tech_id = _extract_external_id(t)
            if not tech_id:
                continue
            for phase in t.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") == "mitre-attack":
                    tactic_shortname = phase.get("phase_name", "")
                    relations.append({
                        "tech_id": tech_id,
                        "tactic_shortname": tactic_shortname,
                    })

        query = """
        UNWIND $batch AS item
        MATCH (tech:Technique {attack_id: item.tech_id})
        MATCH (tac:Tactic {shortname: item.tactic_shortname})
        MERGE (tech)-[:BELONGS_TO]->(tac)
        """

        with self.driver.session() as session:
            session.run(query, batch=relations)

        return len(relations)

    def _import_stix_relationships(self, relationships: list, stix_to_attack: dict) -> dict:
        """
        导入 STIX 关系对象
        将 source_ref/target_ref (STIX ID) 映射为 attack_id，创建 Neo4j 关系
        """
        # 要导入的关系类型及其对应的 Neo4j 关系名
        rel_type_map = {
            "uses": "USES",
            "mitigates": "MITIGATES",
            "subtechnique-of": "SUBTECHNIQUE_OF",
            "attributed-to": "ATTRIBUTED_TO",
            "detects": "DETECTS",
        }

        # 按关系类型分组
        grouped = {}
        skipped = 0
        for rel in relationships:
            rel_type = rel.get("relationship_type", "")
            if rel_type not in rel_type_map:
                continue
            source_aid = stix_to_attack.get(rel.get("source_ref", ""))
            target_aid = stix_to_attack.get(rel.get("target_ref", ""))
            if not source_aid or not target_aid:
                skipped += 1
                continue
            neo4j_rel = rel_type_map[rel_type]
            grouped.setdefault(neo4j_rel, []).append({
                "source": source_aid,
                "target": target_aid,
            })

        # 逐类型导入
        counts = {}
        for neo4j_rel, pairs in grouped.items():
            # 分批处理，每批 1000 条
            batch_size = 1000
            total_imported = 0
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                # 用 UNWIND + MERGE 导入，source/target 可能是多种节点类型
                # 用通用的 MATCH，不限定节点标签
                query = f"""
                UNWIND $batch AS item
                MATCH (src {{attack_id: item.source}})
                MATCH (tgt {{attack_id: item.target}})
                MERGE (src)-[r:{neo4j_rel}]->(tgt)
                """
                with self.driver.session() as session:
                    session.run(query, batch=batch)
                total_imported += len(batch)
            counts[neo4j_rel] = total_imported
            print(f"  {neo4j_rel}: {total_imported} 条")

        if skipped > 0:
            print(f"  跳过 {skipped} 条无法映射的关系")

        return counts

    def build(self):
        """
        主构建流程：解析数据 → 创建约束 → 批量导入节点 → 建立关系
        """
        print("=" * 50)
        print("开始构建 MITRE ATT&CK 知识图谱")
        print("=" * 50)

        # 1. 解析数据
        data = self._parse_attack_data()

        # 2. 创建约束
        print("\n创建唯一约束...")
        self._create_constraints()

        # 3. 导入所有节点
        print("\n--- 导入节点 ---")

        print("\n导入战术节点...")
        imported = self._import_tactics(data["tactics"])
        print(f"已导入 {imported}/{len(data['tactics'])} 个战术")

        print("\n导入技术节点...")
        imported = self._import_techniques(data["techniques"])
        print(f"已导入 {imported}/{len(data['techniques'])} 个技术")

        print("\n导入威胁组织节点...")
        imported = self._import_groups(data["groups"])
        print(f"已导入 {imported}/{len(data['groups'])} 个威胁组织")

        print("\n导入恶意软件节点...")
        imported = self._import_malware(data["malware"])
        print(f"已导入 {imported}/{len(data['malware'])} 个恶意软件")

        print("\n导入工具节点...")
        imported = self._import_tools(data["tools"])
        print(f"已导入 {imported}/{len(data['tools'])} 个工具")

        print("\n导入缓解措施节点...")
        imported = self._import_mitigations(data["mitigations"])
        print(f"已导入 {imported}/{len(data['mitigations'])} 个缓解措施")

        print("\n导入攻击活动节点...")
        imported = self._import_campaigns(data["campaigns"])
        print(f"已导入 {imported}/{len(data['campaigns'])} 个攻击活动")

        # 4. 建立关系
        print("\n--- 建立关系 ---")

        print("\n建立技术-战术关系 (BELONGS_TO)...")
        rel_count = self._import_technique_tactic_relations(data["techniques"])
        print(f"已创建 {rel_count} 条 BELONGS_TO 关系")

        print("\n导入 STIX 关系 (USES/MITIGATES/SUBTECHNIQUE_OF/ATTRIBUTED_TO/DETECTS)...")
        counts = self._import_stix_relationships(
            data["relationships"], data["stix_to_attack"]
        )

        # 5. 汇总
        print("\n" + "=" * 50)
        print("✅ 知识图谱构建完成！")
        total_rels = sum(counts.values()) + rel_count
        print(f"  节点: 战术 {len(data['tactics'])} | 技术 {len(data['techniques'])} | "
              f"组织 {len(data['groups'])} | 恶意软件 {len(data['malware'])} | "
              f"工具 {len(data['tools'])} | 缓解 {len(data['mitigations'])} | "
              f"活动 {len(data['campaigns'])}")
        print(f"  关系: BELONGS_TO {rel_count} | " +
              " | ".join(f"{k} {v}" for k, v in counts.items()))
        print(f"  总关系: {total_rels}")
        print("=" * 50)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="MITRE ATT&CK 知识图谱构建工具")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Neo4j 连接地址 (默认: bolt://localhost:7687)")
    parser.add_argument("--user", default=DEFAULT_USER, help="Neo4j 用户名 (默认: neo4j)")
    parser.add_argument("--password", default="", help="Neo4j 密码（或设置 NEO4J_PASSWORD 环境变量）")
    args = parser.parse_args()

    password = args.password or os.environ.get("NEO4J_PASSWORD", "")

    builder = None
    try:
        builder = GraphBuilder(uri=args.uri, user=args.user, password=password)
        builder.build()
    except (ConnectionError, FileNotFoundError, ValueError) as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if builder:
            builder.close()


if __name__ == "__main__":
    main()
