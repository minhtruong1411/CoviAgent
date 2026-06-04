import os
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph
graph = Neo4jGraph(url=os.environ.get('NEO4J_URI'), username=os.environ.get('NEO4J_USERNAME'), password=os.environ.get('NEO4J_PASSWORD'), database=os.environ.get('NEO4J_DATABASE'))
res = graph.query("MATCH (c:Chunk) WHERE c.text CONTAINS 'antigen tests, and provide low- or no-co' RETURN c.text LIMIT 1")
print('NEO4J MATCH:', len(res))
