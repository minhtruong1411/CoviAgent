import os
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph
graph = Neo4jGraph(url=os.environ.get('NEO4J_URI'), username=os.environ.get('NEO4J_USERNAME'), password=os.environ.get('NEO4J_PASSWORD'), database=os.environ.get('NEO4J_DATABASE'))

# Qdrant chunk text
qdrant_text = "antigen tests, and provide low- or no-cost testing. Free NAAT or antigen tests may also be available through your local health department. \u2022 Talk to a doctor or healthcare provider about other testing options that may be available to you. \u2022 If you are a person with a disability, the Disability Information and Access Line can help you access a test or find a test location. Interpreting your results If your COVID-19 test is positive A positive COVID-19 test means the virus was detected and you have or recently had an infection. \u2022 Take steps to prevent spreading COVID-19. \u2022 Monitor your symptoms. If you have any emergency warning signs, seek emergency care immediately. \u2022 Seek health care right away for treatment if you have risk factors for severe illness. Treatment may be an option to make your symptoms less severe and shorten the time you are sick. Treatment needs to be started within a few days of when your symptoms begin. If your COVID-19 test is negative A negative COVID-19 test means the test did not detect the virus, but this doesn't rule out that you could have an infection. If you used an antigen test, follow FDA recommendations for repeat testing. \u2022 If you have symptoms: o You may have COVID-19 but tested before the virus was detectable. o You may have another viral infection or illness. o Take actions to help protect yourself and others from health risks caused by respiratory viruses. o Contact a healthcare provider if you have any questions about your test result."

words = [w for w in qdrant_text.split() if len(w) > 5]
snippet_clauses = [f"toLower(c.text) CONTAINS '{w.lower().replace(chr(39), '')}'" for w in words[:3]]
clause = " AND ".join(snippet_clauses)

print('CYPHER:', clause)

cypher = f"MATCH (c:Chunk) WHERE {clause} RETURN c.text LIMIT 1"
res = graph.query(cypher)
print('NEO4J MATCH:', len(res))
if res:
    print('FOUND:', res[0]['c.text'][:100])
