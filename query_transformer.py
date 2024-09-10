from langchain_groq import ChatGroq
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from dotenv import load_dotenv
import os

LLAMA3 = "llama3-70b-8192"
MIXTRAL = "mixtral-8x7b-32768"

CUSTOM_INSTRUCTION_FILE = "instructions/c4dsl.txt"

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types, node labels and properties in the schema.
Do not use any other relationship types, node labels or properties that are not provided.
Schema:
{schema}
Notes: 
Do not use the size() function, prefer COUNT instead
For searching part of strings prefer CONTAINS or STARTS WITH function over regular expressions, e.g. prefer  c:Class WHERE c.name CONTAINS "xyz"
The return type of method is defined as (:METHOD)-[:RETURN_TYPE]->(:TYPE)
The relation between a field and its type is (:FIELD)-[:OF_TYPE]->(:TYPE)


Examples: Here are a few examples of generated Cypher statements for particular questions:
# Show me all classes that declare less than 3 public methods
MATCH (c:Class)-[:DECLARES_METHOD]->(m:Method)
WHERE m.accessModifier = "public"
WITH c, COUNT(m) AS methodCount
WHERE methodCount < 3
RETURN c.name

The question is:
{question}"""

def initialize_keys():

    print("initialize_keys")

    global groq_api_key, NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD  # Declare the use of global variables

    # Initialize the global variables
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY') or st.secrets.get("GROQ_API_KEY")
    NEO4J_URL = os.getenv('NEO4J_URL') or st.secrets.get("NEO4J_URL")
    NEO4J_USER = os.getenv('NEO4J_USER') or st.secrets.get("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD') or st.secrets.get("NEO4J_PASSWORD")

def get_cypher_llm():
    return ChatGroq(temperature=0,model_name=LLAMA3)

def get_qa_llm():
    return ChatGroq(temperature=0,model_name=MIXTRAL)

def get_prompt():
    return PromptTemplate(
        input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
    )

def get_custom_instructions():
    with open(CUSTOM_INSTRUCTION_FILE, 'r') as file:
        return file.read()

def get_graph():
    graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASSWORD, refresh_schema=True)
    print(graph.schema)
    return graph

def get_query_chain():
    return GraphCypherQAChain.from_llm(
        cypher_llm=get_cypher_llm(), qa_llm=get_qa_llm(), graph=get_graph(), verbose=True, cypher_prompt=get_prompt(), return_intermediate_steps=True
    )

def invoke_with_query(query: str, chain: GraphCypherQAChain):
    return chain.invoke( {'query': {query}})
