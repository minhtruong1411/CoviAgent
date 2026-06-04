import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

# Load biến môi trường từ file .env
load_dotenv()

# Kết nối với Neo4j
graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD")
)

# Xóa dữ liệu cũ (cẩn thận nếu bạn đang dùng database có sẵn)
graph.query("MATCH (n) DETACH DELETE n")

# Tạo dữ liệu y khoa mẫu: Bệnh (Disease), Triệu chứng (Symptom), Điều trị (Treatment)
cypher_query = """
CREATE (d1:Disease {name: 'Bệnh tiểu đường Type 2', description: 'Rối loạn chuyển hóa mạn tính'})
CREATE (d2:Disease {name: 'Cảm cúm', description: 'Bệnh truyền nhiễm do virus hô hấp'})

CREATE (s1:Symptom {name: 'Khát nước liên tục'})
CREATE (s2:Symptom {name: 'Đi tiểu nhiều'})
CREATE (s3:Symptom {name: 'Sốt cao'})
CREATE (s4:Symptom {name: 'Đau đầu'})

CREATE (t1:Treatment {name: 'Metformin', type: 'Thuốc uống'})
CREATE (t2:Treatment {name: 'Nghỉ ngơi và uống nhiều nước', type: 'Chăm sóc tại nhà'})
CREATE (t3:Treatment {name: 'Paracetamol', type: 'Thuốc giảm đau hạ sốt'})

MERGE (d1)-[:HAS_SYMPTOM]->(s1)
MERGE (d1)-[:HAS_SYMPTOM]->(s2)
MERGE (d1)-[:RECOMMENDS_TREATMENT]->(t1)

MERGE (d2)-[:HAS_SYMPTOM]->(s3)
MERGE (d2)-[:HAS_SYMPTOM]->(s4)
MERGE (d2)-[:RECOMMENDS_TREATMENT]->(t2)
MERGE (d2)-[:RECOMMENDS_TREATMENT]->(t3)
"""

graph.query(cypher_query)
print("Đã khởi tạo xong dữ liệu Knowledge Graph y khoa!")