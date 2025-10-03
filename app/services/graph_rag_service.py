import json
import logging
import os
import random
import re
import sys
import textwrap
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

# ✅ New import paths (per langchain-neo4j)
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from neo4j import GraphDatabase
from rapidfuzz import process

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.graph_RAG import GraphRAGService
from src.llm.api import init_genai_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedGraphRAGService:
    def __init__(self):
        self.config = self._load_config()
        # ✅ Use Neo4jGraph from langchain-neo4j (supports database arg)
        self.kg = Neo4jGraph(
            url=self.config["neo4j_uri"],
            username=self.config["neo4j_username"],
            password=self.config["neo4j_password"],
            database=self.config["neo4j_database"],
        )
        self.graph_rag = GraphRAGService()
        self.cypher_chain = self._setup_cypher_chain()
        self.node_mapping, self.edge_list = self._load_graph_data()
        # Delay expensive embedding computation until actually needed
        self.embeddings = None

    def _load_config(self) -> Dict[str, str]:
        return {
            "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_username": os.getenv("NEO4J_USERNAME", "neo4j"),
            "neo4j_password": os.getenv("NEO4J_PASSWORD", "neo4jpassword"),
            "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
            "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        }

    def _setup_cypher_chain(self):
        if not self.config["google_api_key"]:
            raise ValueError("GOOGLE_API_KEY not set")

        template = self._get_cypher_template()
        prompt = PromptTemplate(
            input_variables=["schema", "question"], template=template
        )

        # Ensure API key is available to the Google GenAI client
        os.environ["GOOGLE_API_KEY"] = self.config["google_api_key"]

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.3,
        )

        # ✅ GraphCypherQAChain from langchain-neo4j
        return GraphCypherQAChain.from_llm(
            llm=llm,
            graph=self.kg,
            cypher_prompt=prompt,
            top_k=20,
            verbose=True,
            allow_dangerous_requests=True,
        )

    def _get_cypher_template(self) -> str:
        return """
Role: Generate exactly ONE Cypher statement for the given question.

Hard constraints:
- Phân tích câu hỏi, suy ra thực thể và quan hệ nhưng CHỈ dùng nhãn (labels), loại quan hệ (relationship types) và thuộc tính (properties) có trong {schema}.
- KHÔNG tự tạo nhãn/quan hệ/thuộc tính mới; KHÔNG dùng APOC; KHÔNG thêm chú thích/giải thích.
- So khớp tên không phân biệt hoa/thường và giữ dấu tiếng Việt:
  Ví dụ: WHERE toLower(node.name) = toLower('Giá trị')

Schema:
{schema}

Mơ hồ & giới hạn:
- Nếu câu hỏi mơ hồ về loại quan hệ, chỉ chọn trong các loại có thật trong schema.
- Với yêu cầu “tổng hợp cấu hình/thông số”, WHITELIST các quan hệ đặc tả/phổ biến (nếu tồn tại trong schema), ví dụ:
  type(r) IN [
    'CÓ_THÔNG_SỐ','CÓ_ĐẶC_TÍNH','CÓ_GIÁ_TRỊ','CÓ_GIÁ_TRỊ_CHO',
    'ĐƯỢC_LẮP_ĐẶT_TẠI','LẮP_TRONG','LÀM_NƠI_LẮP_ĐẶT',
    'KẾT_NỐI_VỚI','KẾT_NỐI_QUA','KẾT_NỐI_BẰNG','TRUYỀN_DỮ_LIỆU_QUA','TRUYỀN_DỮ_LIỆU_TỚI','TRUYỀN_DỮ_LIỆU_ĐẾN','TRUYỀN','TRUYỀN_THÔNG_TIN_TỚI','LÀ_ĐIỂM_KẾT_NỐI',
    'ĐƯỢC_SẢN_XUẤT_BỞI','CÓ_XUẤT_XỨ_TỪ','TUÂN_THỦ','ĐÁP_ỨNG_TIÊU_CHUẨN','ĐẠT_TIÊU_CHUẨN','CÓ_CHỨNG_CHỈ',
    'BAO_GỒM','TÍCH_HỢP','TÍCH_HỢP_VÀO','SỬ_DỤNG','SỬ_DỤNG_TRONG','SỬ_DỤNG_CÔNG_NGHỆ','ỨNG_DỤNG','ỨNG_DỤNG_CHO','ỨNG_DỤNG_TẠI','GIÁM_SÁT',
    'HỖ_TRỢ','HỖ_TRỢ_GIAO_THỨC','CÓ_MODEL','ĐO'
  ]
- Hạn chế kết quả bằng LIMIT phù hợp (ví dụ 25) trừ khi câu hỏi đòi 1 giá trị duy nhất.

Output format:
- CHỈ in ra 1 câu lệnh Cypher hợp lệ. KHÔNG in thêm bất kỳ văn bản nào khác.

Patterns (chỉ dùng nếu tồn tại trong schema):

# 1) Thiết bị được lắp ở đâu?
# Hỏi: "LS-BE-001 được lắp đặt ở đâu?"
MATCH (d:Entity)-[:ĐƯỢC_LẮP_ĐẶT_TẠI]->(loc:Entity)
  WHERE toLower(d.name) = toLower('LS-BE-001')
RETURN loc.name

# 2) Thiết bị nào đo một tham số?
# Hỏi: "Thiết bị nào đo Tốc độ gió?"
MATCH (dev:Entity)-[:ĐO]->(param:Entity)
  WHERE toLower(param.name) = toLower('Tốc độ gió')
RETURN dev.name
LIMIT 25

# 3) Lấy giá trị của một thông số cụ thể của thiết bị
# Hỏi: "Ngõ giao tiếp dữ liệu của LS-BE-001 là gì?"
MATCH (d:Entity)-[:CÓ_THÔNG_SỐ]->(spec:Entity)-[:CÓ_GIÁ_TRỊ]->(val:Entity)
  WHERE toLower(d.name) = toLower('LS-BE-001')
    AND toLower(spec.name) = toLower('Ngõ giao tiếp dữ liệu')
RETURN val.name

# 4) Thiết bị tuân thủ tiêu chuẩn nào?
# Hỏi: "LS-BE-001 tuân thủ tiêu chuẩn nào?"
MATCH (d:Entity)-[:TUÂN_THỦ]->(std:Entity)
  WHERE toLower(d.name) = toLower('LS-BE-001')
RETURN std.name
LIMIT 25

# 5) Thiết bị nào được lắp tại một vị trí cụ thể?
# Hỏi: "Thiết bị nào được lắp tại Mép cầu cảng?"
MATCH (dev:Entity)-[:ĐƯỢC_LẮP_ĐẶT_TẠI]->(loc:Entity)
  WHERE toLower(loc.name) = toLower('Mép cầu cảng')
RETURN dev.name
LIMIT 25

# 6) Thiết bị kết nối với thành phần nào?
# Hỏi: "Thiết bị nào kết nối với Datalogger trung tâm (mục 5.1)?"
MATCH (dev:Entity)-[:KẾT_NỐI_VỚI]->(target:Entity)
  WHERE toLower(target.name) = toLower('Datalogger trung tâm (mục 5.1')
RETURN dev.name
LIMIT 25

# 7) Lấy model của một thiết bị
# Hỏi: "Model của Marine Wind Monitor là gì?"
MATCH (d:Entity)-[:CÓ_MODEL]->(m:Entity)
  WHERE toLower(d.name) = toLower('Marine Wind Monitor')
RETURN m.name

# 8) Liệt kê “cấu hình/thông số” tổng hợp cho một thiết bị
# Hỏi: "Cho tôi cấu hình của LS-BE-001?"
MATCH (a:Entity)-[r]->(b:Entity)
  WHERE toLower(a.name) = toLower('LS-BE-001')
    AND type(r) IN [
      'CÓ_THÔNG_SỐ','CÓ_ĐẶC_TÍNH','CÓ_GIÁ_TRỊ','CÓ_GIÁ_TRỊ_CHO',
      'ĐƯỢC_LẮP_ĐẶT_TẠI','LẮP_TRONG','LÀM_NƠI_LẮP_ĐẶT',
      'KẾT_NỐI_VỚI','KẾT_NỐI_QUA','KẾT_NỐI_BẰNG',
      'ĐƯỢC_SẢN_XUẤT_BỞI','CÓ_XUẤT_XỨ_TỪ','TUÂN_THỦ','ĐÁP_ỨNG_TIÊU_CHUẨN','ĐẠT_TIÊU_CHUẨN','CÓ_CHỨNG_CHỈ'
    ]
RETURN a AS node_a, r AS relationship, b AS node_b
LIMIT 25

The question is:
{question}
"""

    def _load_graph_data(self) -> Tuple[Dict[int, str], List[Tuple[int, int, str]]]:
        try:
            driver = GraphDatabase.driver(
                self.config["neo4j_uri"],
                auth=(self.config["neo4j_username"], self.config["neo4j_password"]),
            )
            with driver.session(database=self.config["neo4j_database"]) as session:
                nodes_query = "MATCH (n:Entity) RETURN id(n) AS node_id, n.name AS name"
                nodes = session.run(nodes_query)
                node_mapping = {record["node_id"]: record["name"] for record in nodes}

                edges_query = "MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS relationship_type"
                edges = session.run(edges_query)
                edge_list = [
                    (record["source"], record["target"], record["relationship_type"])
                    for record in edges
                ]

            logger.info(f"Loaded {len(node_mapping)} nodes and {len(edge_list)} edges")
            return node_mapping, edge_list
        except Exception as e:
            logger.error(f"Error loading graph data: {e}")
            return {}, []

    def _compute_embeddings(self):
        if not self.node_mapping:
            return None

        try:
            import torch  # type: ignore
            import torch.nn.functional as F  # type: ignore
            from torch_geometric.nn import GCNConv  # type: ignore
        except Exception as e:
            logger.warning(f"Torch/torch_geometric not available, skip embeddings: {e}")
            return None

        edge_index = (
            torch.tensor([[e[0], e[1]] for e in self.edge_list], dtype=torch.long)
            .t()
            .contiguous()
        )
        num_nodes = len(self.node_mapping)
        features = torch.eye(num_nodes)

        class GAE(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, embedding_dim):
                super().__init__()
                self.encoder1 = GCNConv(input_dim, hidden_dim)
                self.encoder2 = GCNConv(hidden_dim, embedding_dim)

            def encode(self, x, edge_index):
                x = F.relu(self.encoder1(x, edge_index))
                return self.encoder2(x, edge_index)

        model = GAE(features.size(1), 16, 8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            z = model.encode(features, edge_index)
            loss = torch.mean(z**2)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model.encode(features, edge_index)

        logger.info("Embeddings computed")
        return embeddings

    def find_closest_entities(
        self, entities: List[str]
    ) -> List[Tuple[str, int, str, float]]:
        results = []
        node_names = list(self.node_mapping.values())
        if not node_names:
            return results
        for entity in entities:
            match = process.extractOne(entity, node_names)
            if not match:
                continue
            closest_match, score, index = match
            closest_match_id = list(self.node_mapping.keys())[int(index)]
            results.append((entity, closest_match_id, closest_match, float(score)))
        return results

    def find_similar_nodes(
        self, node_id: int, k: int = 20
    ) -> List[Tuple[int, str, float]]:
        # Compute embeddings on demand to avoid long startup time
        if self.embeddings is None:
            logger.info("Embeddings not yet computed — computing now (lazy)")
            self.embeddings = self._compute_embeddings()
            if self.embeddings is None:
                # If embeddings cannot be computed (missing deps or graph empty), return empty
                return []

        try:
            import torch  # type: ignore
        except Exception:
            return []

        query_embedding = self.embeddings[node_id]
        similarities = torch.matmul(
            query_embedding.unsqueeze(0), self.embeddings.T
        ).squeeze()
        top_k_indices = torch.topk(similarities, k).indices.tolist()
        results: List[Tuple[int, str, float]] = []
        for idx in top_k_indices:
            idx_int = int(idx)
            name = self.node_mapping.get(idx_int, str(idx_int))
            score = float(similarities[idx_int].item())
            results.append((idx_int, name, score))
        return results

    def find_indirect_connection(
        self, start: int, target: int, max_depth: int = 10
    ) -> List[List[Tuple[int, str, int]]]:
        graph = {}
        for src, tgt, rel in self.edge_list:
            graph.setdefault(src, []).append((tgt, rel))
            graph.setdefault(tgt, []).append((src, rel))

        queue = deque([(start, [], 0)])
        visited = set()
        paths = []

        while queue:
            current, path, depth = queue.popleft()
            if depth > max_depth:
                continue
            if current == target:
                paths.append(path)
                continue
            visited.add(current)
            for neighbor, rel in graph.get(current, []):
                if neighbor not in visited:
                    queue.append(
                        (neighbor, path + [(current, rel, neighbor)], depth + 1)
                    )
        return paths

    def process_query(
        self, user_input: str, active_entity: Optional[str] = None
    ) -> str:
        try:
            start_total = time.time()

            # Extract entities (may call LLM) - measure time
            t0 = time.time()
            logger.info("Starting entity extraction")
            entities_result = self.graph_rag.extract_entities_and_relationships(
                user_input
            )
            t1 = time.time()
            logger.info(f"Entity extraction took {t1 - t0:.2f}s")

            # Parse LLM output (may call LLM or do local parsing) - measure time
            t0 = time.time()
            entities, _ = self.graph_rag.parse_llm_output(entities_result)
            t1 = time.time()
            logger.info(f"Parsing LLM output took {t1 - t0:.2f}s")

            if not entities:
                # If no entities extracted, try to use active_entity fallback when provided.
                # This helps follow-up questions like "nó có thông số gì?" to resolve to the
                # previously mentioned entity.
                t0 = time.time()
                logger.info(
                    "No entities extracted — trying active_entity fallback or running cypher_chain.invoke directly"
                )

                query_for_cypher = user_input
                if active_entity:
                    try:
                        # Replace common Vietnamese pronouns that refer to previous entity
                        pronoun_pattern = (
                            r"\b(nó|này|đó|hệ thống này|sản phẩm này|thiết bị này)\b"
                        )
                        replaced = re.sub(
                            pronoun_pattern,
                            active_entity,
                            user_input,
                            flags=re.IGNORECASE,
                        )
                        if replaced != user_input:
                            query_for_cypher = replaced
                        else:
                            # Prepend the active_entity to give context if no pronoun present
                            query_for_cypher = f"{active_entity} {user_input}"
                        logger.info(
                            f"Using active_entity fallback for cypher query: {query_for_cypher}"
                        )
                    except Exception as e:
                        logger.warning(f"Active entity fallback failed: {e}")

                out = self.cypher_chain.invoke({"query": query_for_cypher})
                t1 = time.time()
                logger.info(f"Cypher chain invoke took {t1 - t0:.2f}s")
                logger.info(
                    f"Total process_query time: {time.time() - start_total:.2f}s"
                )
                return out.get("result", str(out))

            # Find matches
            t0 = time.time()
            matches = self.find_closest_entities(entities)
            t1 = time.time()
            logger.info(f"find_closest_entities took {t1 - t0:.2f}s")
            best_matches = [m for m in matches if m[3] > 70]

            if not best_matches:
                return "Không tìm thấy thông tin phù hợp."

            # Enhance query
            enhanced_query = user_input
            for query_entity, _, match_name, _ in best_matches:
                enhanced_query = enhanced_query.replace(query_entity, match_name, 1)

            out = self.cypher_chain.invoke({"query": enhanced_query})
            logger.info(
                f"Cypher invoke completed in {time.time() - start_total:.2f}s total"
            )
            return out.get("result", str(out))
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "Có lỗi xảy ra khi xử lý câu hỏi."


# Usage
if __name__ == "__main__":
    service = OptimizedGraphRAGService()
    response = service.process_query("LS-BE-001 có thông số gì?")
    logger.info("ℹ️ %s", response)
