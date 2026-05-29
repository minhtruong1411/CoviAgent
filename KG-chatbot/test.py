import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Load file .env chứa NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
load_dotenv()

# Kết nối vào AuraDB của bạn
graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
    database=os.environ.get("NEO4J_DATABASE", "neo4j")
)

print("Đang quét cấu trúc database trên AuraDB...\n")
graph.refresh_schema()

# In ra toàn bộ Schema (Bao gồm Node properties và Relationship)
print("========== CẤU TRÚC ĐỒ THỊ (SCHEMA) ==========")
print(graph.schema)
print("==============================================")