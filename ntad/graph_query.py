"""
Neo4j 图查询模块
根据 BM25 命中的 attack_id，用 Cypher 查询 Neo4j 获取关联实体和关系
"""

import os
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def _detect_entity_type(attack_id: str) -> str:
    """根据 attack_id 前缀判断实体类型"""
    if attack_id.startswith("TA"):
        return "tactic"
    if attack_id.startswith("T"):
        return "technique"
    if attack_id.startswith("G"):
        return "group"
    if attack_id.startswith("S"):
        return "software"
    if attack_id.startswith("M"):
        return "mitigation"
    if attack_id.startswith("C"):
        return "campaign"
    return "unknown"


class GraphQuery:
    """Neo4j 图查询器"""

    def __init__(self, uri: str = "", user: str = "", password: str = ""):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")

        if not self.password:
            raise ValueError("请提供 Neo4j 密码（参数或 NEO4J_PASSWORD 环境变量）")

        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            print(f"✅ GraphQuery 已连接 Neo4j: {self.uri}")
        except Exception as e:
            raise ConnectionError(f"Neo4j 连接失败: {e}")

    def close(self):
        if hasattr(self, "driver"):
            self.driver.close()

    def query_related_entities(self, attack_ids: list) -> dict:
        """
        批量查询实体及其关联实体（多跳）
        支持所有实体类型：Technique, Tactic, Group, Software, Mitigation, Campaign

        Args:
            attack_ids: attack_id 列表

        Returns:
            包含所有相关节点和关系的字典
        """
        if not attack_ids:
            return {"techniques": [], "tactics": [], "groups": [], "software": [],
                    "mitigations": [], "campaigns": [], "relationships": []}

        # 按类型分组
        by_type = {}
        for aid in attack_ids:
            t = _detect_entity_type(aid)
            by_type.setdefault(t, []).append(aid)

        techniques = []
        tactics = []
        groups = []
        software = []
        mitigations = []
        campaigns = []
        relationships = []

        # 收集需要二跳查询的 ID
        tac_ids = set()
        group_ids = set()
        sw_ids = set()
        mit_ids = set()
        camp_ids = set()

        with self.driver.session() as session:
            # === 查询命中的 Technique 及其关联 ===
            if by_type.get("technique"):
                query = """
                MATCH (t:Technique) WHERE t.attack_id IN $ids
                OPTIONAL MATCH (t)-[:BELONGS_TO]->(tac:Tactic)
                OPTIONAL MATCH (grp:Group)-[:USES]->(t)
                OPTIONAL MATCH (sw)-[:USES]->(t) WHERE sw:Malware OR sw:Tool
                OPTIONAL MATCH (mit:Mitigation)-[:MITIGATES]->(t)
                OPTIONAL MATCH (camp:Campaign)-[:USES]->(t)
                OPTIONAL MATCH (sub:Technique)-[:SUBTECHNIQUE_OF]->(t)
                RETURN t.attack_id AS aid, t.name AS name, t.description AS desc,
                       t.platforms AS platforms, t.url AS url, 'technique' AS etype,
                       collect(DISTINCT {id: tac.attack_id, n: tac.name}) AS rel_tactics,
                       collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_groups,
                       collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw,
                       collect(DISTINCT {id: mit.attack_id, n: mit.name}) AS rel_mits,
                       collect(DISTINCT {id: camp.attack_id, n: camp.name}) AS rel_camps,
                       collect(DISTINCT {id: sub.attack_id, n: sub.name}) AS rel_subs
                """
                for r in session.run(query, ids=by_type["technique"]):
                    techniques.append({"attack_id": r["aid"], "name": r["name"],
                                       "description": r["desc"], "platforms": r["platforms"], "url": r["url"]})
                    _add_rels(relationships, r["aid"], r["rel_tactics"], "BELONGS_TO", tac_ids)
                    _add_rels(relationships, r["aid"], r["rel_groups"], "USES", group_ids, reverse=True)
                    _add_rels(relationships, r["aid"], r["rel_sw"], "USES", sw_ids, reverse=True)
                    _add_rels(relationships, r["aid"], r["rel_mits"], "MITIGATES", mit_ids, reverse=True)
                    _add_rels(relationships, r["aid"], r["rel_camps"], "USES", camp_ids, reverse=True)
                    # 子技术关系
                    for sub in r["rel_subs"]:
                        if sub["id"]:
                            relationships.append({"from": sub["id"], "to": r["aid"],
                                                  "type": "SUBTECHNIQUE_OF", "to_name": sub["n"]})

            # === 查询命中的 Group 及其关联 ===
            if by_type.get("group"):
                query = """
                MATCH (g:Group) WHERE g.attack_id IN $ids
                OPTIONAL MATCH (g)-[:USES]->(tech:Technique)
                OPTIONAL MATCH (g)-[:USES]->(sw) WHERE sw:Malware OR sw:Tool
                OPTIONAL MATCH (camp:Campaign)-[:ATTRIBUTED_TO]->(g)
                RETURN g.attack_id AS aid, g.name AS name, g.description AS desc,
                       g.aliases AS aliases, g.url AS url, 'group' AS etype,
                       collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                       collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw,
                       collect(DISTINCT {id: camp.attack_id, n: camp.name}) AS rel_camps
                """
                for r in session.run(query, ids=by_type["group"]):
                    groups.append({"attack_id": r["aid"], "name": r["name"],
                                   "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})
                    _add_rels(relationships, r["aid"], r["rel_techs"], "USES", None)
                    for sw in r["rel_sw"]:
                        if sw["id"]:
                            relationships.append({"from": r["aid"], "to": sw["id"], "type": "USES", "to_name": sw["n"]})
                            sw_ids.add(sw["id"])
                    for camp in r["rel_camps"]:
                        if camp["id"]:
                            relationships.append({"from": camp["id"], "to": r["aid"], "type": "ATTRIBUTED_TO", "to_name": camp["n"]})
                            camp_ids.add(camp["id"])

            # === 查询命中的 Software 及其关联 ===
            if by_type.get("software"):
                query = """
                MATCH (sw) WHERE sw.attack_id IN $ids AND (sw:Malware OR sw:Tool)
                OPTIONAL MATCH (sw)-[:USES]->(tech:Technique)
                OPTIONAL MATCH (grp:Group)-[:USES]->(sw)
                RETURN sw.attack_id AS aid, sw.name AS name, sw.description AS desc,
                       sw.platforms AS platforms, sw.aliases AS aliases, sw.url AS url,
                       labels(sw) AS labels,
                       collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                       collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_grps
                """
                for r in session.run(query, ids=by_type["software"]):
                    sw_type = "malware" if "Malware" in r["labels"] else "tool"
                    software.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"],
                                     "platforms": r["platforms"], "aliases": r["aliases"], "url": r["url"], "type": sw_type})
                    for tech in r["rel_techs"]:
                        if tech["id"]:
                            relationships.append({"from": r["aid"], "to": tech["id"], "type": "USES", "to_name": tech["n"]})
                    for grp in r["rel_grps"]:
                        if grp["id"]:
                            relationships.append({"from": grp["id"], "to": r["aid"], "type": "USES", "to_name": grp["n"]})
                            group_ids.add(grp["id"])

            # === 查询命中的 Tactic 及其关联 ===
            if by_type.get("tactic"):
                query = """
                MATCH (tac:Tactic) WHERE tac.attack_id IN $ids
                OPTIONAL MATCH (tech:Technique)-[:BELONGS_TO]->(tac)
                RETURN tac.attack_id AS aid, tac.name AS name, tac.description AS desc,
                       tac.shortname AS shortname, tac.url AS url,
                       collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs
                """
                for r in session.run(query, ids=by_type["tactic"]):
                    tactics.append({"attack_id": r["aid"], "name": r["name"],
                                    "description": r["desc"], "shortname": r["shortname"], "url": r["url"]})
                    for tech in r["rel_techs"]:
                        if tech["id"]:
                            relationships.append({"from": tech["id"], "to": r["aid"], "type": "BELONGS_TO", "to_name": r["name"]})

            # === 查询命中的 Mitigation 及其关联 ===
            if by_type.get("mitigation"):
                query = """
                MATCH (m:Mitigation) WHERE m.attack_id IN $ids
                OPTIONAL MATCH (m)-[:MITIGATES]->(tech:Technique)
                RETURN m.attack_id AS aid, m.name AS name, m.description AS desc, m.url AS url,
                       collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs
                """
                for r in session.run(query, ids=by_type["mitigation"]):
                    mitigations.append({"attack_id": r["aid"], "name": r["name"],
                                        "description": r["desc"], "url": r["url"]})
                    for tech in r["rel_techs"]:
                        if tech["id"]:
                            relationships.append({"from": r["aid"], "to": tech["id"], "type": "MITIGATES", "to_name": tech["n"]})

            # === 查询命中的 Campaign 及其关联 ===
            if by_type.get("campaign"):
                query = """
                MATCH (c:Campaign) WHERE c.attack_id IN $ids
                OPTIONAL MATCH (c)-[:USES]->(tech:Technique)
                OPTIONAL MATCH (c)-[:USES]->(sw) WHERE sw:Malware OR sw:Tool
                OPTIONAL MATCH (c)-[:ATTRIBUTED_TO]->(grp:Group)
                RETURN c.attack_id AS aid, c.name AS name, c.description AS desc,
                       c.aliases AS aliases, c.url AS url,
                       collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                       collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw,
                       collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_grps
                """
                for r in session.run(query, ids=by_type["campaign"]):
                    campaigns.append({"attack_id": r["aid"], "name": r["name"],
                                      "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})
                    for tech in r["rel_techs"]:
                        if tech["id"]:
                            relationships.append({"from": r["aid"], "to": tech["id"], "type": "USES", "to_name": tech["n"]})
                    for sw in r["rel_sw"]:
                        if sw["id"]:
                            relationships.append({"from": r["aid"], "to": sw["id"], "type": "USES", "to_name": sw["n"]})
                            sw_ids.add(sw["id"])
                    for grp in r["rel_grps"]:
                        if grp["id"]:
                            relationships.append({"from": r["aid"], "to": grp["id"], "type": "ATTRIBUTED_TO", "to_name": grp["n"]})
                            group_ids.add(grp["id"])

            # === 二跳查询：补充关联实体的详情 ===
            # 去掉已有实体的 ID
            existing_tech_ids = {t["attack_id"] for t in techniques}
            existing_group_ids = {g["attack_id"] for g in groups}
            existing_sw_ids = {s["attack_id"] for s in software}
            existing_mit_ids = {m["attack_id"] for m in mitigations}
            existing_camp_ids = {c["attack_id"] for c in campaigns}

            # 补充战术详情
            if tac_ids:
                tac_ids -= {t["attack_id"] for t in tactics}
                if tac_ids:
                    for r in session.run("MATCH (t:Tactic) WHERE t.attack_id IN $ids RETURN t.attack_id AS aid, t.name AS name, t.description AS desc, t.shortname AS sn, t.url AS url", ids=list(tac_ids)):
                        tactics.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"], "shortname": r["sn"], "url": r["url"]})

            # 补充组织详情
            new_group_ids = group_ids - existing_group_ids
            if new_group_ids:
                for r in session.run("MATCH (g:Group) WHERE g.attack_id IN $ids RETURN g.attack_id AS aid, g.name AS name, g.description AS desc, g.aliases AS aliases, g.url AS url", ids=list(new_group_ids)):
                    groups.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})

            # 补充软件详情
            new_sw_ids = sw_ids - existing_sw_ids
            if new_sw_ids:
                for r in session.run("MATCH (sw) WHERE sw.attack_id IN $ids AND (sw:Malware OR sw:Tool) RETURN sw.attack_id AS aid, sw.name AS name, sw.description AS desc, sw.platforms AS platforms, sw.aliases AS aliases, sw.url AS url, labels(sw) AS labels", ids=list(new_sw_ids)):
                    sw_type = "malware" if "Malware" in r["labels"] else "tool"
                    software.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"],
                                     "platforms": r["platforms"], "aliases": r["aliases"], "url": r["url"], "type": sw_type})

            # 补充缓解措施详情
            new_mit_ids = mit_ids - existing_mit_ids
            if new_mit_ids:
                for r in session.run("MATCH (m:Mitigation) WHERE m.attack_id IN $ids RETURN m.attack_id AS aid, m.name AS name, m.description AS desc, m.url AS url", ids=list(new_mit_ids)):
                    mitigations.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"], "url": r["url"]})

            # 补充攻击活动详情
            new_camp_ids = camp_ids - existing_camp_ids
            if new_camp_ids:
                for r in session.run("MATCH (c:Campaign) WHERE c.attack_id IN $ids RETURN c.attack_id AS aid, c.name AS name, c.description AS desc, c.aliases AS aliases, c.url AS url", ids=list(new_camp_ids)):
                    campaigns.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})

        return {
            "techniques": techniques,
            "tactics": tactics,
            "groups": groups,
            "software": software,
            "mitigations": mitigations,
            "campaigns": campaigns,
            "relationships": relationships,
        }

    def query_by_intent(self, attack_ids: list, query_focus: str = "general", entities: list = None) -> dict:
        """
        根据用户意图动态决定查询策略

        Args:
            attack_ids: BM25 命中的 attack_id 列表
            query_focus: 用户关心的关系类型
            entities: LLM 提取的原始实体

        Returns:
            同 query_related_entities 的返回格式
        """
        if not attack_ids:
            return {"techniques": [], "tactics": [], "groups": [], "software": [],
                    "mitigations": [], "campaigns": [], "relationships": []}

        by_type = {}
        for aid in attack_ids:
            t = _detect_entity_type(aid)
            by_type.setdefault(t, []).append(aid)

        techniques, tactics, groups, software = [], [], [], []
        mitigations, campaigns, relationships = [], [], []

        with self.driver.session() as session:
            if query_focus == "detail":
                self._query_detail(session, by_type, techniques, tactics, groups,
                                   software, mitigations, campaigns, relationships)
            elif query_focus == "uses":
                self._query_uses(session, by_type, techniques, groups, software,
                                 campaigns, relationships)
            elif query_focus == "mitigates":
                self._query_mitigates(session, by_type, techniques, mitigations, relationships)
            elif query_focus == "belongs_to":
                self._query_belongs_to(session, by_type, techniques, tactics, relationships)
            elif query_focus == "attributed_to":
                self._query_attributed_to(session, by_type, campaigns, groups, relationships)
            elif query_focus == "tactics_of":
                self._query_tactics_of(session, by_type, tactics, techniques, relationships)
            else:
                self._query_general(session, by_type, techniques, tactics, groups,
                                    software, mitigations, campaigns, relationships)

        return {
            "techniques": techniques, "tactics": tactics, "groups": groups,
            "software": software, "mitigations": mitigations, "campaigns": campaigns,
            "relationships": relationships,
        }

    def _query_detail(self, session, by_type, techniques, tactics, groups,
                      software, mitigations, campaigns, relationships):
        """查实体本身详情 + 所有直接关系（1跳）"""
        if by_type.get("technique"):
            q = """
            MATCH (t:Technique) WHERE t.attack_id IN $ids
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(tac:Tactic)
            OPTIONAL MATCH (grp:Group)-[:USES]->(t)
            OPTIONAL MATCH (sw)-[:USES]->(t) WHERE sw:Malware OR sw:Tool
            OPTIONAL MATCH (mit:Mitigation)-[:MITIGATES]->(t)
            OPTIONAL MATCH (camp:Campaign)-[:USES]->(t)
            OPTIONAL MATCH (sub:Technique)-[:SUBTECHNIQUE_OF]->(t)
            RETURN t.attack_id AS aid, t.name AS name, t.description AS desc,
                   t.platforms AS platforms, t.url AS url,
                   collect(DISTINCT {id: tac.attack_id, n: tac.name}) AS rel_tactics,
                   collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_groups,
                   collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw,
                   collect(DISTINCT {id: mit.attack_id, n: mit.name}) AS rel_mits,
                   collect(DISTINCT {id: camp.attack_id, n: camp.name}) AS rel_camps,
                   collect(DISTINCT {id: sub.attack_id, n: sub.name}) AS rel_subs
            """
            tac_ids, group_ids, sw_ids, mit_ids, camp_ids = set(), set(), set(), set(), set()
            for r in session.run(q, ids=by_type["technique"]):
                techniques.append({"attack_id": r["aid"], "name": r["name"],
                                   "description": r["desc"], "platforms": r["platforms"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_tactics"], "BELONGS_TO", tac_ids)
                _add_rels(relationships, r["aid"], r["rel_groups"], "USES", group_ids, reverse=True)
                _add_rels(relationships, r["aid"], r["rel_sw"], "USES", sw_ids, reverse=True)
                _add_rels(relationships, r["aid"], r["rel_mits"], "MITIGATES", mit_ids, reverse=True)
                _add_rels(relationships, r["aid"], r["rel_camps"], "USES", camp_ids, reverse=True)
                for sub in r["rel_subs"]:
                    if sub["id"]:
                        relationships.append({"from": sub["id"], "to": r["aid"],
                                              "type": "SUBTECHNIQUE_OF", "to_name": sub["n"]})
            self._fill_2hop(session, techniques, tactics, groups, software, mitigations, campaigns,
                            tac_ids, group_ids, sw_ids, mit_ids, camp_ids)

        if by_type.get("group"):
            q = """
            MATCH (g:Group) WHERE g.attack_id IN $ids
            OPTIONAL MATCH (g)-[:USES]->(tech:Technique)
            OPTIONAL MATCH (g)-[:USES]->(sw) WHERE sw:Malware OR sw:Tool
            RETURN g.attack_id AS aid, g.name AS name, g.description AS desc,
                   g.aliases AS aliases, g.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                   collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw
            """
            for r in session.run(q, ids=by_type["group"]):
                groups.append({"attack_id": r["aid"], "name": r["name"],
                               "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "USES", None)
                _add_rels(relationships, r["aid"], r["rel_sw"], "USES", None)

        if by_type.get("software"):
            q = """
            MATCH (sw) WHERE sw.attack_id IN $ids AND (sw:Malware OR sw:Tool)
            OPTIONAL MATCH (sw)-[:USES]->(tech:Technique)
            OPTIONAL MATCH (grp:Group)-[:USES]->(sw)
            RETURN sw.attack_id AS aid, sw.name AS name, sw.description AS desc,
                   sw.platforms AS platforms, sw.aliases AS aliases, sw.url AS url,
                   labels(sw) AS labels,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                   collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_grps
            """
            for r in session.run(q, ids=by_type["software"]):
                sw_type = "malware" if "Malware" in r["labels"] else "tool"
                software.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"],
                                 "platforms": r["platforms"], "aliases": r["aliases"], "url": r["url"], "type": sw_type})
                _add_rels(relationships, r["aid"], r["rel_techs"], "USES", None)
                _add_rels(relationships, r["aid"], r["rel_grps"], "USES", None, reverse=True)

        if by_type.get("tactic"):
            q = """
            MATCH (tac:Tactic) WHERE tac.attack_id IN $ids
            OPTIONAL MATCH (tech:Technique)-[:BELONGS_TO]->(tac)
            RETURN tac.attack_id AS aid, tac.name AS name, tac.description AS desc,
                   tac.shortname AS shortname, tac.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs
            """
            for r in session.run(q, ids=by_type["tactic"]):
                tactics.append({"attack_id": r["aid"], "name": r["name"],
                                "description": r["desc"], "shortname": r["shortname"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "BELONGS_TO", None, reverse=True)

        if by_type.get("mitigation"):
            q = """
            MATCH (m:Mitigation) WHERE m.attack_id IN $ids
            OPTIONAL MATCH (m)-[:MITIGATES]->(tech:Technique)
            RETURN m.attack_id AS aid, m.name AS name, m.description AS desc, m.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs
            """
            for r in session.run(q, ids=by_type["mitigation"]):
                mitigations.append({"attack_id": r["aid"], "name": r["name"],
                                    "description": r["desc"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "MITIGATES", None)

        if by_type.get("campaign"):
            q = """
            MATCH (c:Campaign) WHERE c.attack_id IN $ids
            OPTIONAL MATCH (c)-[:USES]->(tech:Technique)
            OPTIONAL MATCH (c)-[:ATTRIBUTED_TO]->(grp:Group)
            RETURN c.attack_id AS aid, c.name AS name, c.description AS desc,
                   c.aliases AS aliases, c.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                   collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_grps
            """
            for r in session.run(q, ids=by_type["campaign"]):
                campaigns.append({"attack_id": r["aid"], "name": r["name"],
                                  "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "USES", None)
                _add_rels(relationships, r["aid"], r["rel_grps"], "ATTRIBUTED_TO", None)

    def _query_uses(self, session, by_type, techniques, groups, software, campaigns, relationships):
        """只查 USES 关系"""
        if by_type.get("technique"):
            q = """
            MATCH (t:Technique) WHERE t.attack_id IN $ids
            OPTIONAL MATCH (grp:Group)-[:USES]->(t)
            OPTIONAL MATCH (sw)-[:USES]->(t) WHERE sw:Malware OR sw:Tool
            OPTIONAL MATCH (sub:Technique)-[:SUBTECHNIQUE_OF]->(t)
            RETURN t.attack_id AS aid, t.name AS name, t.description AS desc,
                   t.platforms AS platforms, t.url AS url,
                   collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_groups,
                   collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw,
                   collect(DISTINCT {id: sub.attack_id, n: sub.name}) AS rel_subs
            """
            group_ids, sw_ids = set(), set()
            for r in session.run(q, ids=by_type["technique"]):
                techniques.append({"attack_id": r["aid"], "name": r["name"],
                                   "description": r["desc"], "platforms": r["platforms"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_groups"], "USES", group_ids, reverse=True)
                _add_rels(relationships, r["aid"], r["rel_sw"], "USES", sw_ids, reverse=True)
                for sub in r["rel_subs"]:
                    if sub["id"]:
                        relationships.append({"from": sub["id"], "to": r["aid"],
                                              "type": "SUBTECHNIQUE_OF", "to_name": sub["n"]})
            self._fill_2hop(session, [], [], groups, software, [], [],
                            set(), group_ids, sw_ids, set(), set())

        if by_type.get("group"):
            q = """
            MATCH (g:Group) WHERE g.attack_id IN $ids
            OPTIONAL MATCH (g)-[:USES]->(tech:Technique)
            OPTIONAL MATCH (g)-[:USES]->(sw) WHERE sw:Malware OR sw:Tool
            RETURN g.attack_id AS aid, g.name AS name, g.description AS desc,
                   g.aliases AS aliases, g.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                   collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw
            """
            tech_ids, sw_ids = set(), set()
            for r in session.run(q, ids=by_type["group"]):
                groups.append({"attack_id": r["aid"], "name": r["name"],
                               "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "USES", tech_ids)
                _add_rels(relationships, r["aid"], r["rel_sw"], "USES", sw_ids)
            if tech_ids:
                for r in session.run(
                    "MATCH (t:Technique) WHERE t.attack_id IN $ids "
                    "RETURN t.attack_id AS aid, t.name AS name, t.description AS desc, "
                    "t.platforms AS platforms, t.url AS url",
                    ids=list(tech_ids)):
                    techniques.append({"attack_id": r["aid"], "name": r["name"],
                                       "description": r["desc"], "platforms": r["platforms"], "url": r["url"]})
            if sw_ids:
                for r in session.run(
                    "MATCH (sw) WHERE sw.attack_id IN $ids AND (sw:Malware OR sw:Tool) "
                    "RETURN sw.attack_id AS aid, sw.name AS name, sw.description AS desc, "
                    "sw.platforms AS platforms, sw.aliases AS aliases, sw.url AS url, labels(sw) AS labels",
                    ids=list(sw_ids)):
                    sw_type = "malware" if "Malware" in r["labels"] else "tool"
                    software.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"],
                                     "platforms": r["platforms"], "aliases": r["aliases"], "url": r["url"], "type": sw_type})

        if by_type.get("software"):
            q = """
            MATCH (sw) WHERE sw.attack_id IN $ids AND (sw:Malware OR sw:Tool)
            OPTIONAL MATCH (sw)-[:USES]->(tech:Technique)
            OPTIONAL MATCH (grp:Group)-[:USES]->(sw)
            RETURN sw.attack_id AS aid, sw.name AS name, sw.description AS desc,
                   sw.platforms AS platforms, sw.aliases AS aliases, sw.url AS url,
                   labels(sw) AS labels,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                   collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_grps
            """
            group_ids = set()
            for r in session.run(q, ids=by_type["software"]):
                sw_type = "malware" if "Malware" in r["labels"] else "tool"
                software.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"],
                                 "platforms": r["platforms"], "aliases": r["aliases"], "url": r["url"], "type": sw_type})
                _add_rels(relationships, r["aid"], r["rel_techs"], "USES", None)
                _add_rels(relationships, r["aid"], r["rel_grps"], "USES", group_ids, reverse=True)
            if group_ids:
                for r in session.run(
                    "MATCH (g:Group) WHERE g.attack_id IN $ids "
                    "RETURN g.attack_id AS aid, g.name AS name, g.description AS desc, g.aliases AS aliases, g.url AS url",
                    ids=list(group_ids)):
                    groups.append({"attack_id": r["aid"], "name": r["name"],
                                   "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})

        if by_type.get("campaign"):
            q = """
            MATCH (c:Campaign) WHERE c.attack_id IN $ids
            OPTIONAL MATCH (c)-[:USES]->(tech:Technique)
            OPTIONAL MATCH (c)-[:USES]->(sw) WHERE sw:Malware OR sw:Tool
            RETURN c.attack_id AS aid, c.name AS name, c.description AS desc,
                   c.aliases AS aliases, c.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                   collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw
            """
            for r in session.run(q, ids=by_type["campaign"]):
                campaigns.append({"attack_id": r["aid"], "name": r["name"],
                                  "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "USES", None)
                _add_rels(relationships, r["aid"], r["rel_sw"], "USES", None)

    def _query_mitigates(self, session, by_type, techniques, mitigations, relationships):
        """只查 MITIGATES 关系"""
        if by_type.get("technique"):
            q = """
            MATCH (t:Technique) WHERE t.attack_id IN $ids
            OPTIONAL MATCH (mit:Mitigation)-[:MITIGATES]->(t)
            RETURN t.attack_id AS aid, t.name AS name, t.description AS desc,
                   t.platforms AS platforms, t.url AS url,
                   collect(DISTINCT {id: mit.attack_id, n: mit.name}) AS rel_mits
            """
            mit_ids = set()
            for r in session.run(q, ids=by_type["technique"]):
                techniques.append({"attack_id": r["aid"], "name": r["name"],
                                   "description": r["desc"], "platforms": r["platforms"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_mits"], "MITIGATES", mit_ids, reverse=True)
            if mit_ids:
                for r in session.run(
                    "MATCH (m:Mitigation) WHERE m.attack_id IN $ids "
                    "RETURN m.attack_id AS aid, m.name AS name, m.description AS desc, m.url AS url",
                    ids=list(mit_ids)):
                    mitigations.append({"attack_id": r["aid"], "name": r["name"],
                                        "description": r["desc"], "url": r["url"]})

        if by_type.get("mitigation"):
            q = """
            MATCH (m:Mitigation) WHERE m.attack_id IN $ids
            OPTIONAL MATCH (m)-[:MITIGATES]->(tech:Technique)
            RETURN m.attack_id AS aid, m.name AS name, m.description AS desc, m.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs
            """
            for r in session.run(q, ids=by_type["mitigation"]):
                mitigations.append({"attack_id": r["aid"], "name": r["name"],
                                    "description": r["desc"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "MITIGATES", None)

    def _query_belongs_to(self, session, by_type, techniques, tactics, relationships):
        """只查 BELONGS_TO 关系（技术→战术）"""
        if by_type.get("technique"):
            q = """
            MATCH (t:Technique) WHERE t.attack_id IN $ids
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(tac:Tactic)
            RETURN t.attack_id AS aid, t.name AS name, t.description AS desc,
                   t.platforms AS platforms, t.url AS url,
                   collect(DISTINCT {id: tac.attack_id, n: tac.name}) AS rel_tactics
            """
            tac_ids = set()
            for r in session.run(q, ids=by_type["technique"]):
                techniques.append({"attack_id": r["aid"], "name": r["name"],
                                   "description": r["desc"], "platforms": r["platforms"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_tactics"], "BELONGS_TO", tac_ids)
            if tac_ids:
                for r in session.run(
                    "MATCH (t:Tactic) WHERE t.attack_id IN $ids "
                    "RETURN t.attack_id AS aid, t.name AS name, t.description AS desc, t.shortname AS sn, t.url AS url",
                    ids=list(tac_ids)):
                    tactics.append({"attack_id": r["aid"], "name": r["name"],
                                    "description": r["desc"], "shortname": r["sn"], "url": r["url"]})

    def _query_attributed_to(self, session, by_type, campaigns, groups, relationships):
        """只查 ATTRIBUTED_TO 关系（活动→组织）"""
        if by_type.get("campaign"):
            q = """
            MATCH (c:Campaign) WHERE c.attack_id IN $ids
            OPTIONAL MATCH (c)-[:ATTRIBUTED_TO]->(grp:Group)
            RETURN c.attack_id AS aid, c.name AS name, c.description AS desc,
                   c.aliases AS aliases, c.url AS url,
                   collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_grps
            """
            group_ids = set()
            for r in session.run(q, ids=by_type["campaign"]):
                campaigns.append({"attack_id": r["aid"], "name": r["name"],
                                  "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_grps"], "ATTRIBUTED_TO", group_ids)
            if group_ids:
                for r in session.run(
                    "MATCH (g:Group) WHERE g.attack_id IN $ids "
                    "RETURN g.attack_id AS aid, g.name AS name, g.description AS desc, g.aliases AS aliases, g.url AS url",
                    ids=list(group_ids)):
                    groups.append({"attack_id": r["aid"], "name": r["name"],
                                   "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})

    def _query_tactics_of(self, session, by_type, tactics, techniques, relationships):
        """从战术出发查其下属技术（BELONGS_TO 反向）"""
        if by_type.get("tactic"):
            q = """
            MATCH (tac:Tactic) WHERE tac.attack_id IN $ids
            OPTIONAL MATCH (tech:Technique)-[:BELONGS_TO]->(tac)
            RETURN tac.attack_id AS aid, tac.name AS name, tac.description AS desc,
                   tac.shortname AS shortname, tac.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs
            """
            for r in session.run(q, ids=by_type["tactic"]):
                tactics.append({"attack_id": r["aid"], "name": r["name"],
                                "description": r["desc"], "shortname": r["shortname"], "url": r["url"]})
                for tech in r["rel_techs"]:
                    if tech["id"]:
                        techniques.append({"attack_id": tech["id"], "name": tech["n"],
                                           "description": "", "platforms": [], "url": ""})
                        relationships.append({"from": tech["id"], "to": r["aid"],
                                              "type": "BELONGS_TO", "to_name": r["name"]})

    def _query_general(self, session, by_type, techniques, tactics, groups,
                       software, mitigations, campaigns, relationships):
        """通用查询：所有关系，1跳"""
        if by_type.get("technique"):
            q = """
            MATCH (t:Technique) WHERE t.attack_id IN $ids
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(tac:Tactic)
            OPTIONAL MATCH (grp:Group)-[:USES]->(t)
            OPTIONAL MATCH (sw)-[:USES]->(t) WHERE sw:Malware OR sw:Tool
            OPTIONAL MATCH (mit:Mitigation)-[:MITIGATES]->(t)
            OPTIONAL MATCH (camp:Campaign)-[:USES]->(t)
            OPTIONAL MATCH (sub:Technique)-[:SUBTECHNIQUE_OF]->(t)
            RETURN t.attack_id AS aid, t.name AS name, t.description AS desc,
                   t.platforms AS platforms, t.url AS url,
                   collect(DISTINCT {id: tac.attack_id, n: tac.name}) AS rel_tactics,
                   collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_groups,
                   collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw,
                   collect(DISTINCT {id: mit.attack_id, n: mit.name}) AS rel_mits,
                   collect(DISTINCT {id: camp.attack_id, n: camp.name}) AS rel_camps,
                   collect(DISTINCT {id: sub.attack_id, n: sub.name}) AS rel_subs
            """
            tac_ids, group_ids, sw_ids, mit_ids, camp_ids = set(), set(), set(), set(), set()
            for r in session.run(q, ids=by_type["technique"]):
                techniques.append({"attack_id": r["aid"], "name": r["name"],
                                   "description": r["desc"], "platforms": r["platforms"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_tactics"], "BELONGS_TO", tac_ids)
                _add_rels(relationships, r["aid"], r["rel_groups"], "USES", group_ids, reverse=True)
                _add_rels(relationships, r["aid"], r["rel_sw"], "USES", sw_ids, reverse=True)
                _add_rels(relationships, r["aid"], r["rel_mits"], "MITIGATES", mit_ids, reverse=True)
                _add_rels(relationships, r["aid"], r["rel_camps"], "USES", camp_ids, reverse=True)
                for sub in r["rel_subs"]:
                    if sub["id"]:
                        relationships.append({"from": sub["id"], "to": r["aid"],
                                              "type": "SUBTECHNIQUE_OF", "to_name": sub["n"]})
            self._fill_2hop(session, techniques, tactics, groups, software, mitigations, campaigns,
                            tac_ids, group_ids, sw_ids, mit_ids, camp_ids)

        if by_type.get("group"):
            q = """
            MATCH (g:Group) WHERE g.attack_id IN $ids
            OPTIONAL MATCH (g)-[:USES]->(tech:Technique)
            OPTIONAL MATCH (g)-[:USES]->(sw) WHERE sw:Malware OR sw:Tool
            OPTIONAL MATCH (camp:Campaign)-[:ATTRIBUTED_TO]->(g)
            RETURN g.attack_id AS aid, g.name AS name, g.description AS desc,
                   g.aliases AS aliases, g.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                   collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw,
                   collect(DISTINCT {id: camp.attack_id, n: camp.name}) AS rel_camps
            """
            sw_ids, camp_ids = set(), set()
            for r in session.run(q, ids=by_type["group"]):
                groups.append({"attack_id": r["aid"], "name": r["name"],
                               "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "USES", None)
                _add_rels(relationships, r["aid"], r["rel_sw"], "USES", sw_ids)
                _add_rels(relationships, r["aid"], r["rel_camps"], "ATTRIBUTED_TO", camp_ids, reverse=True)

        if by_type.get("software"):
            q = """
            MATCH (sw) WHERE sw.attack_id IN $ids AND (sw:Malware OR sw:Tool)
            OPTIONAL MATCH (sw)-[:USES]->(tech:Technique)
            OPTIONAL MATCH (grp:Group)-[:USES]->(sw)
            RETURN sw.attack_id AS aid, sw.name AS name, sw.description AS desc,
                   sw.platforms AS platforms, sw.aliases AS aliases, sw.url AS url,
                   labels(sw) AS labels,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                   collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_grps
            """
            for r in session.run(q, ids=by_type["software"]):
                sw_type = "malware" if "Malware" in r["labels"] else "tool"
                software.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"],
                                 "platforms": r["platforms"], "aliases": r["aliases"], "url": r["url"], "type": sw_type})
                _add_rels(relationships, r["aid"], r["rel_techs"], "USES", None)
                _add_rels(relationships, r["aid"], r["rel_grps"], "USES", None, reverse=True)

        if by_type.get("tactic"):
            q = """
            MATCH (tac:Tactic) WHERE tac.attack_id IN $ids
            OPTIONAL MATCH (tech:Technique)-[:BELONGS_TO]->(tac)
            RETURN tac.attack_id AS aid, tac.name AS name, tac.description AS desc,
                   tac.shortname AS shortname, tac.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs
            """
            for r in session.run(q, ids=by_type["tactic"]):
                tactics.append({"attack_id": r["aid"], "name": r["name"],
                                "description": r["desc"], "shortname": r["shortname"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "BELONGS_TO", None, reverse=True)

        if by_type.get("mitigation"):
            q = """
            MATCH (m:Mitigation) WHERE m.attack_id IN $ids
            OPTIONAL MATCH (m)-[:MITIGATES]->(tech:Technique)
            RETURN m.attack_id AS aid, m.name AS name, m.description AS desc, m.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs
            """
            for r in session.run(q, ids=by_type["mitigation"]):
                mitigations.append({"attack_id": r["aid"], "name": r["name"],
                                    "description": r["desc"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "MITIGATES", None)

        if by_type.get("campaign"):
            q = """
            MATCH (c:Campaign) WHERE c.attack_id IN $ids
            OPTIONAL MATCH (c)-[:USES]->(tech:Technique)
            OPTIONAL MATCH (c)-[:USES]->(sw) WHERE sw:Malware OR sw:Tool
            OPTIONAL MATCH (c)-[:ATTRIBUTED_TO]->(grp:Group)
            RETURN c.attack_id AS aid, c.name AS name, c.description AS desc,
                   c.aliases AS aliases, c.url AS url,
                   collect(DISTINCT {id: tech.attack_id, n: tech.name}) AS rel_techs,
                   collect(DISTINCT {id: sw.attack_id, n: sw.name}) AS rel_sw,
                   collect(DISTINCT {id: grp.attack_id, n: grp.name}) AS rel_grps
            """
            sw_ids, group_ids = set(), set()
            for r in session.run(q, ids=by_type["campaign"]):
                campaigns.append({"attack_id": r["aid"], "name": r["name"],
                                  "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})
                _add_rels(relationships, r["aid"], r["rel_techs"], "USES", None)
                _add_rels(relationships, r["aid"], r["rel_sw"], "USES", sw_ids)
                _add_rels(relationships, r["aid"], r["rel_grps"], "ATTRIBUTED_TO", group_ids)

    def _fill_2hop(self, session, techniques, tactics, groups, software,
                   mitigations, campaigns, tac_ids, group_ids, sw_ids, mit_ids, camp_ids):
        """补充二跳关联实体的详情"""
        existing_tech = {t["attack_id"] for t in techniques}
        existing_group = {g["attack_id"] for g in groups}
        existing_sw = {s["attack_id"] for s in software}
        existing_mit = {m["attack_id"] for m in mitigations}
        existing_camp = {c["attack_id"] for c in campaigns}

        tac_ids -= {t["attack_id"] for t in tactics}
        if tac_ids:
            for r in session.run(
                "MATCH (t:Tactic) WHERE t.attack_id IN $ids "
                "RETURN t.attack_id AS aid, t.name AS name, t.description AS desc, t.shortname AS sn, t.url AS url",
                ids=list(tac_ids)):
                tactics.append({"attack_id": r["aid"], "name": r["name"],
                                "description": r["desc"], "shortname": r["sn"], "url": r["url"]})

        new_group_ids = group_ids - existing_group
        if new_group_ids:
            for r in session.run(
                "MATCH (g:Group) WHERE g.attack_id IN $ids "
                "RETURN g.attack_id AS aid, g.name AS name, g.description AS desc, g.aliases AS aliases, g.url AS url",
                ids=list(new_group_ids)):
                groups.append({"attack_id": r["aid"], "name": r["name"],
                               "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})

        new_sw_ids = sw_ids - existing_sw
        if new_sw_ids:
            for r in session.run(
                "MATCH (sw) WHERE sw.attack_id IN $ids AND (sw:Malware OR sw:Tool) "
                "RETURN sw.attack_id AS aid, sw.name AS name, sw.description AS desc, "
                "sw.platforms AS platforms, sw.aliases AS aliases, sw.url AS url, labels(sw) AS labels",
                ids=list(new_sw_ids)):
                sw_type = "malware" if "Malware" in r["labels"] else "tool"
                software.append({"attack_id": r["aid"], "name": r["name"], "description": r["desc"],
                                 "platforms": r["platforms"], "aliases": r["aliases"], "url": r["url"], "type": sw_type})

        new_mit_ids = mit_ids - existing_mit
        if new_mit_ids:
            for r in session.run(
                "MATCH (m:Mitigation) WHERE m.attack_id IN $ids "
                "RETURN m.attack_id AS aid, m.name AS name, m.description AS desc, m.url AS url",
                ids=list(new_mit_ids)):
                mitigations.append({"attack_id": r["aid"], "name": r["name"],
                                    "description": r["desc"], "url": r["url"]})

        new_camp_ids = camp_ids - existing_camp
        if new_camp_ids:
            for r in session.run(
                "MATCH (c:Campaign) WHERE c.attack_id IN $ids "
                "RETURN c.attack_id AS aid, c.name AS name, c.description AS desc, c.aliases AS aliases, c.url AS url",
                ids=list(new_camp_ids)):
                campaigns.append({"attack_id": r["aid"], "name": r["name"],
                                  "description": r["desc"], "aliases": r["aliases"], "url": r["url"]})

    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        with self.driver.session() as session:
            stats = {}
            for label in ["Technique", "Tactic", "Group", "Malware", "Tool", "Mitigation", "Campaign"]:
                stats[label.lower() + "s"] = session.run(
                    f"MATCH (n:{label}) RETURN count(n) AS cnt"
                ).single()["cnt"]

            for rel in ["BELONGS_TO", "USES", "MITIGATES", "SUBTECHNIQUE_OF", "ATTRIBUTED_TO"]:
                stats[rel.lower()] = session.run(
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt"
                ).single()["cnt"]

        return stats


def _add_rels(relationships, entity_id, related_list, rel_type, id_set, reverse=False):
    """辅助函数：添加关系到列表并收集关联 ID"""
    for item in related_list:
        if not item["id"]:
            continue
        if reverse:
            relationships.append({"from": item["id"], "to": entity_id, "type": rel_type, "to_name": item["n"]})
        else:
            relationships.append({"from": entity_id, "to": item["id"], "type": rel_type, "to_name": item["n"]})
        if id_set is not None:
            id_set.add(item["id"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j 图查询工具")
    parser.add_argument("--password", default="", help="Neo4j 密码")
    parser.add_argument("--attack-id", default="", help="查询实体 ID")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    args = parser.parse_args()

    gq = GraphQuery(password=args.password)

    if args.stats:
        stats = gq.get_stats()
        print("=== 节点统计 ===")
        for key, val in stats.items():
            if not any(r in key for r in ["belongs", "uses", "mitigates", "subtechnique", "attributed"]):
                print(f"  {key}: {val}")
        print("\n=== 关系统计 ===")
        for key, val in stats.items():
            if any(r in key for r in ["belongs", "uses", "mitigates", "subtechnique", "attributed"]):
                print(f"  {key}: {val}")

    if args.attack_id:
        result = gq.query_related_entities([args.attack_id])
        for key in ["techniques", "tactics", "groups", "software", "mitigations", "campaigns"]:
            items = result[key]
            if items:
                print(f"\n{key}: {len(items)} 个")
                for item in items[:5]:
                    print(f"  - {item['attack_id']} {item['name']}")
        if result["relationships"]:
            print(f"\n关系: {len(result['relationships'])} 条")
            for rel in result["relationships"][:10]:
                print(f"  {rel['from']} → [{rel['type']}] → {rel['to']}")

    gq.close()
