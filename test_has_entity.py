import os
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph
graph = Neo4jGraph(url=os.environ.get('NEO4J_URI'), username=os.environ.get('NEO4J_USERNAME'), password=os.environ.get('NEO4J_PASSWORD'), database=os.environ.get('NEO4J_DATABASE'))
qdrant_text = 'antigen tests, and provide low- or no-cost testing. Free NAAT or antigen tests may also be available through your local health department. Talk to a doctor or healthcare provider about other testing options that may be available to you. If you are a person with a disability'
words = [w for w in qdrant_text.split() if len(w) > 5]
snippet_clauses = []
for w in words[:3]:
    w_clean = w.lower().replace("'", "").replace('"', "")
    snippet_clauses.append(f"toLower(c.text) CONTAINS '{w_clean}'")
clause = ' AND '.join(snippet_clauses)
res = graph.query(f"MATCH (c:Chunk) WHERE {clause} MATCH (c)-[:HAS_ENTITY]->(e:__Entity__) RETURN e.id LIMIT 10")
print('ENTITIES:', res)
