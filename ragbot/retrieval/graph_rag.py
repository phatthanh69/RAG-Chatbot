import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
import networkx as nx
from langchain_neo4j import Neo4jGraph
from pyvis.network import Network

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphRAGService:
    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=self.google_api_key)
        self.kg = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            database=self.neo4j_database,
        )
        logger.info("Connected to Neo4j successfully.")

    def _call_gemini_with_retry(
        self, prompt_text: str, max_retries: int = 3, backoff_factor: float = 2.0
    ) -> str:
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel("gemini-2.5-pro")
                generation_config = genai.GenerationConfig(temperature=1)
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
                ]
                response = model.generate_content(
                    prompt_text,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                if not getattr(response, "candidates", None):
                    logger.warning(f"Response blocked: {response.prompt_feedback}")
                    return f'Response blocked: {getattr(getattr(response, "prompt_feedback", None), "block_reason", "N/A")}'

                return response.text or ""

            except Exception as e:
                logger.warning(f"Gemini call failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor**attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    return f"Failed after {max_retries} attempts: {str(e)}"

    def extract_entities_and_relationships(self, text: str) -> str:
        prompt = (
            "Extract entities (nodes) and their relationships (edges) from the text below. "
            "Entities and relationships MUST be in Vietnamese. "
            "Include all technical specifications, parameters, manufacturers, origins, installation locations, and other relevant details as entities and relationships. "
            "Common relationship types: ĐƯỢC_SẢN_XUẤT_BỞI, CÓ_XUẤT_XỨ_TỪ, ĐƯỢC_LẮP_ĐẶT_TẠI, CÓ_THÔNG_SỐ, ĐO, HỖ_TRỢ, ỨNG_DỤNG, CÓ_GIÁ_TRỊ, etc. "
            "For technical specifications sections, parse tables and create relationships like (Device, CÓ_THÔNG_SỐ, Parameter), (Parameter, CÓ_GIÁ_TRỊ, Value). "
            "Examples: "
            "- (LS-BE-001, CÓ_THÔNG_SỐ, Dải đo khoảng cách tối đa) "
            "- (Dải đo khoảng cách tối đa, CÓ_GIÁ_TRỊ, 0,5  3.000 m) "
            "- (LS-BE-001, ĐƯỢC_SẢN_XUẤT_BỞI, BlueEco) "
            "Format: "
            "Entities: "
            "- {Entity}: {Type} "
            "Relationships: "
            "- ({Entity1}, {RelationshipType}, {Entity2}) "
            f'Text: "{text}"'
        )
        return self._call_gemini_with_retry(prompt)

    def process_sections_parallel(
        self, sections: List[Dict[str, str]], max_workers: int = 4
    ) -> Dict[str, Any]:
        extracted_data = {}

        def process_section(section: Dict[str, str]) -> Tuple[str, Any]:
            heading = section["heading"]
            content = section["content"].strip()
            if not content:
                return heading, {"error": "Empty content"}
            try:
                info = self.extract_entities_and_relationships(content)
                return heading, info
            except Exception as e:
                logger.error(f"Error processing section {heading}: {e}")
                return heading, {"error": str(e)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_section, section) for section in sections
            ]
            for future in as_completed(futures):
                heading, result = future.result()
                extracted_data[heading] = result
                logger.info(f"Processed section: {heading}")

        return extracted_data

    def parse_llm_output(
        self, result: str
    ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        entity_pattern = r"- (.+): (.+)"
        entities = re.findall(entity_pattern, result)
        entity_dict = {
            entity.strip(): entity_type.strip() for entity, entity_type in entities
        }
        entity_list = list(entity_dict.keys())

        relationship_pattern = r"- \(([^,]+), ([^,]+), ([^)]+)\)"
        relationships = re.findall(relationship_pattern, result)
        relationship_list = []
        for subject, relation, object_ in relationships:
            rel = relation.strip().replace(" ", "_").upper()
            relationship_list.append((subject.strip(), rel, object_.strip()))

        logger.info(
            f"Parsed {len(entity_list)} entities and {len(relationship_list)} relationships"
        )
        return entity_list, relationship_list

    def add_relationships_to_neo4j(self, relationships: List[Tuple[str, str, str]]):
        """
        An toàn & nhanh:
        - Validate REL_TYPE theo regex để tránh chèn tuỳ ý.
        - Nhóm theo REL_TYPE và MERGE bằng UNWIND + tham số (node names).
        """
        if not relationships:
            logger.info("No relationships to add.")
            return

        # Validate & group by relation
        rel_regex = re.compile(r"[A-Z][A-Z0-9_]{0,63}$")
        by_rel: Dict[str, List[Dict[str, str]]] = {}
        skipped = 0

        for s, r, o in relationships:
            r = r.strip().upper()
            if not rel_regex.fullmatch(r):
                skipped += 1
                logger.warning(f"Skip unsafe relationship type: {r}")
                continue
            by_rel.setdefault(r, []).append({"s": s.strip(), "o": o.strip()})

        for rel, rows in by_rel.items():
            # Dùng tham số cho node names; REL_TYPE không thể tham số hoá → đã validate ở trên
            cypher = f"""
            UNWIND $rows AS row
            MERGE (a:Entity {{name: row.s}})
            MERGE (b:Entity {{name: row.o}})
            MERGE (a)-[:{rel}]->(b)
            """
            self.kg.query(cypher, params={"rows": rows})

        logger.info(
            f"Relationships added to Neo4j. Groups: {len(by_rel)} | total rows: {sum(len(v) for v in by_rel.values())} | skipped: {skipped}"
        )

    def export_graph_to_json(self, output_file: str = "graph_data.json"):
        nodes_query = "MATCH (n) RETURN id(n) AS node_id, labels(n) AS labels, properties(n) AS properties"
        nodes = self.kg.query(nodes_query)
        edges_query = "MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS relationship, properties(r) AS properties"
        edges = self.kg.query(edges_query)
        graph_data = {"nodes": nodes, "edges": edges}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Graph data exported to {output_file}")

    def visualize_graph(self, max_nodes: int = 500, output_file: str = "graph.html"):
        limit = max_nodes * 2
        data = self.kg.query(
            f"""
            MATCH (a)-[r]->(b)
            RETURN a.name AS node_a, type(r) AS relationship, b.name AS node_b
            LIMIT {limit}
            """
        )
        G = nx.DiGraph()
        for record in data:
            node_a = record.get("node_a")
            node_b = record.get("node_b")
            relationship = record.get("relationship")
            if node_a and node_b and relationship:
                G.add_node(node_a, label=node_a)
                G.add_node(node_b, label=node_b)
                G.add_edge(node_a, node_b, label=relationship)

        if len(G.nodes) > max_nodes:
            sampled_nodes = list(G.nodes)[:max_nodes]
            G = G.subgraph(sampled_nodes).copy()
            logger.warning(
                f"Graph too large, sampled to {max_nodes} nodes for visualization"
            )

        # notebook=False để chạy dạng script
        net = Network(notebook=False, directed=True, height="750px", width="100%")
        net.from_nx(G)
        net.show(output_file)
        logger.info(f"Graph visualization saved to {output_file}")


if __name__ == "__main__":
    service = GraphRAGService()
    service.export_graph_to_json()
    service.visualize_graph()
