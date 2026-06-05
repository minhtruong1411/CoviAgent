import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

graph = Neo4jGraph(
    url=os.environ.get('NEO4J_URI'),
    username=os.environ.get('NEO4J_USERNAME'),
    password=os.environ.get('NEO4J_PASSWORD'),
    database=os.environ.get('NEO4J_DATABASE')
)

print("=== SCHEMA ===")
try:
    graph.refresh_schema()
    print(graph.schema)
except Exception as e:
    print("Could not get schema:", e)

print("\n=== NODE COUNTS ===")
try:
    res = graph.query("MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC")
    for r in res:
        print(f"{r['label']}: {r['count']}")
except Exception as e:
    print("Error:", e)

print("\n=== RELATIONSHIP COUNTS ===")
try:
    res = graph.query("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC")
    for r in res:
        print(f"{r['type']}: {r['count']}")
except Exception as e:
    print("Error:", e)

print("\n=== SAMPLE NODES (up to 3) ===")
try:
    res = graph.query("MATCH (n) RETURN labels(n)[0] as label, properties(n) as props LIMIT 3")
    for r in res:
        print(f"Label: {r['label']} - Props: {r['props']}")
except Exception as e:
    print("Error:", e)
