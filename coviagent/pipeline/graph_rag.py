"""
GraphRAG Pipeline — Entity-driven hybrid retrieval (Vector + Knowledge Graph).

Đây là GraphRAG "đúng chuẩn": không chỉ ghép text chunk như một vector RAG biến
tướng, mà thực sự khai thác cấu trúc đồ thị ngữ nghĩa trong Neo4j.

Luồng xử lý:
1. Vector Search (Qdrant): tìm chunk liên quan ngữ nghĩa → "điểm vào" (entry point).
2. Anchor Entities: xác định các thực thể (__Entity__) gắn với câu hỏi, từ 2 nguồn
   bổ trợ nhau:
     (a) thực thể được nhắc trực tiếp trong câu hỏi, và
     (b) thực thể gắn với chunk mà vector search tìm được (bridge qua text chuẩn hoá,
         vì Qdrant và Neo4j chunk khác id/khoảng trắng).
3. Graph Traversal: từ anchor, đi theo các QUAN HỆ CÓ KIỂU NGỮ NGHĨA
   (CAUSES, TREATS, DETECTS, INDICATES, PREVENTS, INCREASES_RISK, ...) để thu thập
   các "facts" dạng (chủ thể) -[quan hệ]-> (đối tượng) — đây là phần mà vector RAG
   không thể có.
4. Evidence Chunks: lấy thêm các chunk nguồn gắn với những thực thể liên quan vừa
   khám phá (bổ sung chứng cứ mà vector search có thể đã bỏ sót).
5. LLM Generation: ghép [facts đồ thị] + [chunk vector] + [chunk chứng cứ] làm ngữ cảnh.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional, Set

from neo4j import GraphDatabase
import math

from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate, Settings

from coviagent.core.interfaces import BasePipeline, BaseRetriever

logger = logging.getLogger(__name__)

# Quan hệ mang ý nghĩa tri thức (giữ lại để traverse). Bỏ qua các quan hệ
# cấu trúc tài liệu (HAS_ENTITY, PART_OF, NEXT_CHUNK, SIMILAR, FIRST_CHUNK).
SEMANTIC_RELATIONS = [
    "CAUSES", "TREATS", "INDICATES", "PREVENTS",
    "INCREASES_RISK", "REDUCES_RISK", "MAY_REQUIRE", "DETECTS",
]

# Diễn giải quan hệ sang tiếng Việt để LLM đọc ngữ cảnh tự nhiên hơn.
RELATION_VERB_VI = {
    "CAUSES": "gây ra",
    "TREATS": "điều trị",
    "INDICATES": "cho thấy / chỉ báo",
    "PREVENTS": "phòng ngừa",
    "INCREASES_RISK": "làm tăng nguy cơ",
    "REDUCES_RISK": "làm giảm nguy cơ",
    "MAY_REQUIRE": "có thể cần đến",
    "DETECTS": "phát hiện",
}

# Từ khoá trong câu hỏi → các quan hệ nên ưu tiên (intent của người hỏi).
# Dùng để xếp hạng facts: khi entity quá nhiều quan hệ, đưa loại quan hệ đúng ý lên đầu.
INTENT_KEYWORDS = [
    (["triệu chứng", "dấu hiệu", "biểu hiện", "symptom", "sign"], ["CAUSES"]),
    (["xét nghiệm", "chẩn đoán", "phát hiện", "test", "diagnos", "detect"], ["DETECTS"]),
    (["điều trị", "thuốc", "chữa", "treat", "medic", "drug", "therapy"], ["TREATS", "MAY_REQUIRE"]),
    (["phòng", "ngừa", "vaccine", "vắc", "tiêm", "prevent", "protect"], ["PREVENTS", "REDUCES_RISK"]),
    (["nguy cơ", "yếu tố", "rủi ro", "risk", "factor"], ["INCREASES_RISK", "INDICATES"]),
    (["nặng", "mức độ", "nghiêm trọng", "severe", "severity", "complication"], ["INDICATES", "MAY_REQUIRE"]),
]

DEFAULT_GRAPHRAG_PROMPT = (
    "Bạn là một trợ lý y tế chuyên nghiệp về COVID-19.\n\n"
    "Bạn được cung cấp hai loại ngữ cảnh:\n"
    "1. ĐOẠN VĂN THAM KHẢO (NGUỒN CHÍNH): văn bản gốc — DỰA VÀO ĐÂY để viết câu trả lời, "
    "giữ ĐÚNG cấu trúc/cách phân loại/sắc thái của tài liệu (vd nếu tài liệu chia 'thường "
    "gặp / ít gặp / nghiêm trọng' thì trả lời cũng phải theo đúng cách chia đó).\n"
    "2. CHECKLIST TỪ ĐỒ THỊ: danh sách các mục liên quan đã gom nhóm, dùng để bạn KIỂM TRA "
    "xem câu trả lời đã LIỆT KÊ ĐỦ chưa (đồ thị giúp không bỏ sót mục nào).\n\n"
    "NGUYÊN TẮC:\n"
    "- CHỈ dùng thông tin trong ngữ cảnh bên dưới, TUYỆT ĐỐI không bịa.\n"
    "- VIẾT câu trả lời theo ĐOẠN VĂN THAM KHẢO (văn phong + cấu trúc). Dùng CHECKLIST chỉ để "
    "ĐỐI CHIẾU cho đủ ý; KHÔNG sao chép nguyên văn dòng checklist, KHÔNG nhắc tới 'đồ thị'.\n"
    "- Nếu câu hỏi yêu cầu PHÂN NHÓM/PHÂN LOẠI, hãy bám cách phân nhóm trong ĐOẠN VĂN, "
    "rồi dùng checklist để chắc chắn mỗi nhóm đã đủ mục.\n"
    "- Nếu ngữ cảnh không chứa thông tin liên quan, hãy nói: "
    "'Tài liệu hiện tại không có thông tin về vấn đề này.'\n"
    "- Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc, đầy đủ.\n\n"
    "===== ĐOẠN VĂN THAM KHẢO (nguồn chính để viết) =====\n"
    "{context_str}\n"
    "===== CHECKLIST TỪ ĐỒ THỊ (để đối chiếu cho đủ ý) =====\n"
    "{facts_str}\n"
    "==============================\n\n"
    "Câu hỏi: {query_str}\n"
    "Trả lời (viết theo ĐOẠN VĂN, dùng checklist để không bỏ sót ý): "
)

_WORD_RE = re.compile(r"[a-zA-Zà-ỹÀ-Ỹ0-9\-]+", re.UNICODE)


def _normalize(text: str) -> str:
    """Chuẩn hoá: lowercase + gộp mọi khoảng trắng/xuống dòng về 1 dấu cách."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text: str) -> Set[str]:
    """Tách token (>2 ký tự) đã lowercase để so khớp chunk theo độ phủ."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2}


class GraphRAGPipeline(BasePipeline):
    """
    GraphRAG entity-driven: vector tìm điểm vào, đồ thị mở rộng theo quan hệ ngữ nghĩa.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        expand_hops: int = 1,
        max_graph_chunks: int = 5,
        max_facts: int = 40,
        max_anchors: int = 8,
        bridge_threshold: float = 0.35,
        final_chunks: int = 5,
        rerank: bool = True,
        use_entity_grounding: bool = True,
        prompt_template: str = DEFAULT_GRAPHRAG_PROMPT,
    ):
        """
        Args:
            retriever: Vector retriever (Qdrant). top_k của nó = KÍCH THƯỚC POOL ứng viên
                       (nên đặt cao hơn final_chunks, vd 8) để rerank chọn lại.
            llm: LLM sinh câu trả lời.
            neo4j_*: thông tin kết nối Neo4j (fallback về biến môi trường).
            expand_hops: số bước traverse quan hệ ngữ nghĩa (1 hoặc 2).
            max_graph_chunks: số chunk ứng viên tối đa lấy thêm từ đồ thị (entity-grounded
                              + lân cận) để đưa vào pool trước khi rerank.
            max_facts: số facts (triple) tối đa đưa vào ngữ cảnh.
            max_anchors: số anchor entity tối đa.
            bridge_threshold: ngưỡng độ phủ token để nối chunk Qdrant ↔ Neo4j.
            final_chunks: số đoạn prose GIỮ LẠI sau khi rerank (ngân sách ngữ cảnh prose).
            rerank: bật rerank ứng viên theo độ tương đồng embedding với câu hỏi.
            use_entity_grounding: lấy thêm chunk mô tả trực tiếp các entity được hỏi.
            prompt_template: template prompt sinh câu trả lời.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt = PromptTemplate(prompt_template)
        self.expand_hops = max(1, min(2, expand_hops))
        self.max_graph_chunks = max_graph_chunks
        self.max_facts = max_facts
        self.max_anchors = max_anchors
        self.bridge_threshold = bridge_threshold
        self.final_chunks = final_chunks
        self.rerank = rerank
        self.use_entity_grounding = use_entity_grounding

        uri = neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        username = neo4j_username or os.environ.get("NEO4J_USERNAME", "neo4j")
        password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "password")
        self.neo4j_database = neo4j_database or os.environ.get("NEO4J_DATABASE", "neo4j")

        self.neo4j_driver = GraphDatabase.driver(uri, auth=(username, password))

        # Cache để tránh truy vấn lặp lại: danh sách chunk (id + token) và entity ids.
        self._chunk_cache: List[Dict[str, Any]] = []   # [{id, tokens:set}]
        self._chunk_entities: Dict[str, List[str]] = {}  # chunk_id -> [entity_id]
        self._entity_norms: List[Dict[str, str]] = []    # [{id, norm}]

        try:
            self.neo4j_driver.verify_connectivity()
            self._load_graph_cache()
            logger.info(
                "GraphRAG: Neo4j connected. Cached %d chunks, %d entities.",
                len(self._chunk_cache), len(self._entity_norms),
            )
        except Exception as e:
            logger.warning(
                "GraphRAG: Neo4j connection/cache failed (%s). "
                "Graph expansion disabled, falling back to vector-only.", e,
            )
            self.neo4j_driver = None

    # ──────────────────────────────────────────────
    #  Khởi tạo cache đồ thị (chạy 1 lần lúc init)
    # ──────────────────────────────────────────────
    def _load_graph_cache(self) -> None:
        with self.neo4j_driver.session(database=self.neo4j_database) as session:
            # Chunk + entity gắn kèm
            res = session.run(
                "MATCH (c:Chunk) "
                "OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__) "
                "RETURN c.id AS cid, c.text AS text, collect(DISTINCT e.id) AS ents"
            )
            for r in res:
                cid, text, ents = r["cid"], r["text"], [x for x in r["ents"] if x]
                if not cid or not text:
                    continue
                self._chunk_cache.append({"id": cid, "tokens": _tokens(text)})
                self._chunk_entities[cid] = ents

            # Danh sách entity id để dò trong câu hỏi
            res = session.run("MATCH (e:__Entity__) WHERE e.id IS NOT NULL RETURN DISTINCT e.id AS id")
            for r in res:
                eid = r["id"]
                self._entity_norms.append({"id": eid, "norm": _normalize(eid)})

    # ──────────────────────────────────────────────
    #  Bước 2: xác định anchor entities
    # ──────────────────────────────────────────────
    def _bridge_chunk_entities(self, vector_texts: List[str]) -> List[str]:
        """Nối mỗi chunk vector (Qdrant) tới chunk Neo4j gần nhất theo độ phủ token,
        rồi lấy các entity gắn với chunk đó."""
        anchors: List[str] = []
        for vt in vector_texts:
            vtoks = _tokens(vt)
            if not vtoks:
                continue
            best_id, best_score = None, 0.0
            for ch in self._chunk_cache:
                ctoks = ch["tokens"]
                if not ctoks:
                    continue
                # độ phủ token của chunk vector bởi chunk Neo4j
                score = len(vtoks & ctoks) / len(vtoks)
                if score > best_score:
                    best_id, best_score = ch["id"], score
            if best_id and best_score >= self.bridge_threshold:
                anchors.extend(self._chunk_entities.get(best_id, []))
        return anchors

    def _question_entities(self, question: str) -> List[str]:
        """Tìm các entity được nhắc trực tiếp trong câu hỏi (vd: Covid-19, Paxlovid)."""
        qn = _normalize(question)
        hits = []
        for ent in self._entity_norms:
            norm = ent["norm"]
            # khớp cả cụm; chấp nhận entity >= 4 ký tự để tránh nhiễu
            if len(norm) >= 4 and norm in qn:
                hits.append(ent["id"])
        return hits

    def _anchor_entities(self, question: str, vector_texts: List[str]):
        """Trả về (anchors, question_entity_set).

        Entity nhắc trực tiếp trong câu hỏi được ưu tiên tuyệt đối; entity suy ra từ
        chunk vector chỉ bổ sung (giới hạn số lượng) để không làm loãng trọng tâm.
        """
        q_ents = self._question_entities(question)
        bridged = self._bridge_chunk_entities(vector_texts)

        ordered: List[str] = []
        seen: Set[str] = set()
        for eid in q_ents:
            if eid not in seen:
                seen.add(eid)
                ordered.append(eid)
        # chỉ thêm tối đa một nửa hạn mức từ nguồn bridge (kém chắc chắn hơn)
        bridge_budget = max(2, self.max_anchors // 2)
        for eid in bridged:
            if len(ordered) >= self.max_anchors:
                break
            if eid not in seen and bridge_budget > 0:
                seen.add(eid)
                ordered.append(eid)
                bridge_budget -= 1
        return ordered[: self.max_anchors], set(q_ents)

    # ──────────────────────────────────────────────
    #  Bước 3 + 4: traverse quan hệ ngữ nghĩa + lấy chunk chứng cứ
    # ──────────────────────────────────────────────
    @staticmethod
    def _intent_relations(question: str) -> List[str]:
        """Suy ra loại quan hệ nên ưu tiên từ từ khoá trong câu hỏi."""
        qn = _normalize(question)
        prefs: List[str] = []
        for keywords, rels in INTENT_KEYWORDS:
            if any(k in qn for k in keywords):
                for r in rels:
                    if r not in prefs:
                        prefs.append(r)
        return prefs

    def _graph_traverse(self, anchor_ids: List[str], question: str, q_entities: Set[str]):
        """
        Trả về (facts, related_entity_ids).
        facts: list dict {subj, rel, obj, desc}, đã xếp hạng theo độ liên quan câu hỏi.
        """
        facts: List[Dict[str, Any]] = []
        related: Set[str] = set()
        if not anchor_ids:
            return facts, related

        preferred = self._intent_relations(question)
        anchor_set = set(anchor_ids)

        # Lấy triple quanh anchor theo quan hệ ngữ nghĩa, giữ đúng chiều bằng startNode().
        if self.expand_hops == 1:
            cypher = """
                MATCH (a:__Entity__)-[r]-(b:__Entity__)
                WHERE a.id IN $anchors AND type(r) IN $rels
                RETURN DISTINCT
                  CASE WHEN startNode(r) = a THEN a.id ELSE b.id END AS subj,
                  type(r) AS rel,
                  CASE WHEN startNode(r) = a THEN b.id ELSE a.id END AS obj,
                  r.description AS desc
                LIMIT $limit
            """
        else:
            # 2 hop: anchor → trung gian → đối tượng
            cypher = """
                MATCH p = (a:__Entity__)-[rels*1..2]-(b:__Entity__)
                WHERE a.id IN $anchors
                  AND ALL(rel IN relationships(p) WHERE type(rel) IN $rels)
                  AND a.id <> b.id
                WITH relationships(p) AS rs
                UNWIND rs AS r
                RETURN DISTINCT
                  startNode(r).id AS subj,
                  type(r) AS rel,
                  endNode(r).id AS obj,
                  r.description AS desc
                LIMIT $limit
            """

        # Lấy DƯ facts (gấp ~5 lần) rồi xếp hạng + cắt trong Python, để các quan hệ
        # đúng intent không bị LIMIT của Cypher cắt mất khi anchor có quá nhiều cạnh.
        raw: List[Dict[str, Any]] = []
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                res = session.run(
                    cypher,
                    anchors=anchor_ids,
                    rels=SEMANTIC_RELATIONS,
                    limit=max(self.max_facts * 5, 200),
                )
                for r in res:
                    subj, rel, obj = r["subj"], r["rel"], r["obj"]
                    if not subj or not obj or subj == obj:
                        continue
                    raw.append({"subj": subj, "rel": rel, "obj": obj, "desc": r.get("desc")})
        except Exception as e:
            logger.warning("GraphRAG traverse failed: %s", e)
            return facts, related

        def rank_key(f: Dict[str, Any]):
            # điểm càng nhỏ càng ưu tiên (đứng trước)
            rel_pref = preferred.index(f["rel"]) if f["rel"] in preferred else len(preferred) + 1
            # fact gắn trực tiếp tới entity được hỏi thì quan trọng hơn
            touches_q = 0 if (f["subj"] in q_entities or f["obj"] in q_entities) else 1
            # fact gắn tới anchor cũng tốt hơn fact chỉ liên quan gián tiếp
            touches_anchor = 0 if (f["subj"] in anchor_set or f["obj"] in anchor_set) else 1
            has_desc = 0 if f.get("desc") else 1
            return (touches_q, rel_pref, touches_anchor, has_desc)

        raw.sort(key=rank_key)
        for f in raw[: self.max_facts]:
            facts.append(f)
            related.add(f["subj"])
            related.add(f["obj"])

        return facts, related

    def _entity_grounded_chunks(self, entity_ids: List[str], seen_norm_texts: Set[str],
                                limit: int) -> List[str]:
        """Lấy các chunk PROSE mô tả trực tiếp những entity cho trước, xếp hạng theo
        SỐ entity mà chunk đề cập (chunk nhắc tới NHIỀU entity được hỏi → quan trọng hơn).

        Đây là chìa khoá cho câu so sánh/đa thực thể (vd 'Paxlovid vs Lagevrio'):
        chunk nào nhắc cả hai thuốc sẽ được kéo lên đầu, dù vector search bỏ lỡ.
        """
        if not entity_ids:
            return []
        out: List[str] = []
        try:
            cypher = """
                MATCH (c:Chunk)-[:HAS_ENTITY]->(e:__Entity__)
                WHERE e.id IN $ents
                WITH c, count(DISTINCT e) AS overlap
                ORDER BY overlap DESC
                RETURN c.text AS text
                LIMIT $limit
            """
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                res = session.run(cypher, ents=list(entity_ids), limit=limit * 3)
                for r in res:
                    text = r["text"]
                    if not text or _normalize(text) in seen_norm_texts:
                        continue
                    out.append(text)
                    seen_norm_texts.add(_normalize(text))
                    if len(out) >= limit:
                        break
        except Exception as e:
            logger.warning("GraphRAG entity-grounded fetch failed: %s", e)
        return out

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def _rerank_by_query(self, question: str, texts: List[str], top_n: int) -> List[str]:
        """Rerank ứng viên theo độ tương đồng embedding với câu hỏi, giữ top_n.

        Nhờ đó các đoạn prose do đồ thị khơi ra được CẠNH TRANH công bằng với đoạn
        vector; đoạn liên quan nhất luôn được giữ, bất kể đến từ nguồn nào.
        """
        if len(texts) <= top_n:
            return texts
        embed = getattr(Settings, "embed_model", None)
        if not self.rerank or embed is None:
            return texts[:top_n]
        try:
            qv = embed.get_query_embedding(question)
            scored = [(self._cosine(qv, embed.get_text_embedding(t)), i, t)
                      for i, t in enumerate(texts)]
            # điểm cao trước; giữ thứ tự gốc khi hoà (ổn định)
            scored.sort(key=lambda x: (-x[0], x[1]))
            return [t for _, _, t in scored[:top_n]]
        except Exception as e:
            logger.warning("GraphRAG rerank failed (%s); fallback cắt theo thứ tự.", e)
            return texts[:top_n]

    # từ chung chung bị loại khi tính "khái niệm" của một entity (để gộp biến thể trùng)
    _GENERIC_ENT_TOKENS = {"covid-19", "covid", "19", "sars-cov-2", "các", "the", "of", "and"}

    @classmethod
    def _concept_key(cls, name: str) -> frozenset:
        """Khoá khái niệm của một entity: tập tiền tố 4 ký tự của các token có nghĩa.
        Giúp gộp 'Vaccination' / 'Covid-19 Vaccine' / 'Covid-19 Vaccines' về cùng một."""
        toks = [w for w in _WORD_RE.findall(name.lower())
                if w not in cls._GENERIC_ENT_TOKENS and len(w) > 2]
        return frozenset(w[:4] for w in toks) or frozenset([name.lower()])

    @classmethod
    def _format_facts(cls, facts: List[Dict[str, Any]], max_lines: int = 12) -> str:
        """Trình bày facts dạng CHECKLIST CÓ CẤU TRÚC (gom theo quan hệ) thay vì list phẳng.

        Với mỗi loại quan hệ, GOM về phía 'hub' (bên ít giá trị phân biệt) để mỗi dòng
        liệt kê trọn vẹn một nhóm — đây chính là thế mạnh completeness của đồ thị:
          • Covid-19 — gây ra: Fever, Cough, Tiredness, Sore Throat, ...
          • làm tăng nguy cơ Severe Illness: Older Adults, Diabetes, Cancer, ...
          • phát hiện Covid-19: Pcr Tests, Antigen Tests, Naat
        Cách này giúp LLM kiểm tra đủ ý mà KHÔNG bị rối như list phẳng từng triple.
        """
        if not facts:
            return "(Không có gợi ý liên quan.)"

        # khử trùng triple theo khái niệm (gộp biến thể tên)
        uniq, seen = [], set()
        for f in facts:
            key = (f["rel"], cls._concept_key(f["subj"]), cls._concept_key(f["obj"]))
            if key not in seen:
                seen.add(key)
                uniq.append(f)

        # gom theo loại quan hệ, giữ thứ tự xuất hiện (đã xếp hạng theo intent)
        rel_order, by_rel = [], {}
        for f in uniq:
            if f["rel"] not in by_rel:
                by_rel[f["rel"]] = []
                rel_order.append(f["rel"])
            by_rel[f["rel"]].append(f)

        def dedup_concepts(names):
            out, ks = [], set()
            for n in names:
                k = cls._concept_key(n)
                if k not in ks:
                    ks.add(k)
                    out.append(n)
            return out

        lines = []
        for rel in rel_order:
            group = by_rel[rel]
            verb = RELATION_VERB_VI.get(rel, rel.lower())
            subs = [f["subj"] for f in group]
            objs = [f["obj"] for f in group]
            n_subj = len({cls._concept_key(s) for s in subs})
            n_obj = len({cls._concept_key(o) for o in objs})
            # gom về phía có ÍT giá trị phân biệt hơn (phía 'hub')
            if n_obj <= n_subj:
                # nhiều chủ thể → 1 đối tượng: "verb obj: subj1, subj2, ..."
                for obj in dedup_concepts(objs):
                    members = dedup_concepts([f["subj"] for f in group
                                             if cls._concept_key(f["obj"]) == cls._concept_key(obj)])
                    lines.append(f"• {verb} {obj}: " + ", ".join(members))
            else:
                # 1 chủ thể → nhiều đối tượng: "subj verb: obj1, obj2, ..."
                for subj in dedup_concepts(subs):
                    members = dedup_concepts([f["obj"] for f in group
                                             if cls._concept_key(f["subj"]) == cls._concept_key(subj)])
                    lines.append(f"• {subj} {verb}: " + ", ".join(members))
            if len(lines) >= max_lines:
                break
        return "\n".join(lines[:max_lines])

    # ──────────────────────────────────────────────
    #  Dựng ngữ cảnh (KHÔNG gọi LLM) — dùng chung cho query() và thống kê parity
    # ──────────────────────────────────────────────
    def build_context(self, question: str) -> Dict[str, Any]:
        """Hybrid retrieval (vector + đồ thị) + rerank, KHÔNG gọi LLM.

        Quy trình nâng cao:
          1. Vector retrieval lấy POOL ứng viên (top_k của retriever).
          2. Anchor entities (từ câu hỏi + chunk vector).
          3. Traverse quan hệ ngữ nghĩa → facts (cho câu logic/liệt kê).
          4. Entity-grounded: kéo chunk mô tả trực tiếp các anchor (cho câu prose/so sánh).
          5. Gộp POOL = vector ∪ entity-grounded ∪ lân cận, khử trùng.
          6. Rerank POOL theo câu hỏi → giữ final_chunks đoạn prose tốt nhất.
        """
        # Bước 1: Vector retrieval → pool ứng viên
        retrieval_contexts = self.retriever.retrieve(question)
        vector_texts = [ctx.text for ctx in retrieval_contexts]
        seen_norm = {_normalize(t) for t in vector_texts}

        anchors: List[str] = []
        facts: List[Dict[str, Any]] = []
        entity_chunks: List[str] = []
        neighbor_chunks: List[str] = []

        if self.neo4j_driver:
            # Bước 2: anchor entities
            anchors, q_entities = self._anchor_entities(question, vector_texts)
            # Bước 3: facts theo quan hệ ngữ nghĩa
            facts, related = self._graph_traverse(anchors, question, q_entities)
            # Bước 4: entity-grounded — chunk mô tả trực tiếp entity được hỏi (ưu tiên cao)
            if self.use_entity_grounding and anchors:
                entity_chunks = self._entity_grounded_chunks(anchors, seen_norm, self.max_graph_chunks)
            # Bước 4b: chunk lân cận theo các entity khám phá thêm (bổ sung độ phủ)
            extra_ents = [e for e in related if e not in set(anchors)]
            if extra_ents:
                neighbor_chunks = self._entity_grounded_chunks(extra_ents, seen_norm, self.max_graph_chunks)

        # Bước 5: gộp POOL ứng viên (đã khử trùng qua seen_norm ở các bước trên)
        graph_candidates = entity_chunks + neighbor_chunks
        candidate_pool = list(vector_texts) + graph_candidates

        # Bước 6: rerank POOL theo câu hỏi → giữ final_chunks đoạn tốt nhất
        prose_texts = self._rerank_by_query(question, candidate_pool, self.final_chunks)

        # đoạn nào trong kết quả KHÔNG phải từ vector → coi là đóng góp của đồ thị
        vector_norm = {_normalize(t) for t in vector_texts}
        graph_evidence = [t for t in prose_texts if _normalize(t) not in vector_norm]

        facts_str = self._format_facts(facts)
        return {
            "vector_texts": vector_texts,
            "evidence_texts": graph_evidence,
            "prose_texts": prose_texts,
            "facts": facts,
            "facts_str": facts_str,
            "anchors": anchors,
        }

    # ──────────────────────────────────────────────
    #  Pipeline chính
    # ──────────────────────────────────────────────
    def query(self, question: str) -> Dict[str, Any]:
        ctx = self.build_context(question)
        prose_texts = ctx["prose_texts"]
        facts = ctx["facts"]
        facts_str = ctx["facts_str"]

        # Bước 5: ghép ngữ cảnh + sinh câu trả lời
        context_str = "\n\n".join(prose_texts) if prose_texts else "(trống)"
        formatted_prompt = self.prompt.format(
            facts_str=facts_str,
            context_str=context_str,
            query_str=question,
        )
        response = self.llm.complete(formatted_prompt)
        answer = str(response)

        # contexts: gộp cả facts (dạng text) để evaluator/judge thấy nguồn đồ thị.
        all_contexts = ([facts_str] if facts else []) + prose_texts

        return {
            "answer": answer,
            "contexts": all_contexts,
            "vector_contexts": ctx["vector_texts"],
            "graph_facts": [f"{f['subj']} -[{f['rel']}]-> {f['obj']}" for f in facts],
            "graph_evidence_chunks": ctx["evidence_texts"],
            "anchor_entities": ctx["anchors"],
        }

    def __del__(self):
        if hasattr(self, "neo4j_driver") and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except Exception:
                pass
