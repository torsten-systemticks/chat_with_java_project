from langchain_groq import ChatGroq 
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from examples import QUERY_EXAMPLES
import streamlit as st 
from langchain_ollama import ChatOllama


# Constants for LLM models and instruction file
LLAMA3 = "llama3-70b-8192"
MIXTRAL = "mixtral-8x7b-32768"

CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types, node labels, and properties in the schema.
Do not use any other relationship types, node labels, or properties that are not provided.
Schema:
{schema}
Notes: 
Do not use the size() function; prefer COUNT instead.
For searching part of strings prefer CONTAINS or STARTS WITH function over regular expressions.
You must always use the right direction of a relationship.

Examples: Here are a few examples of generated Cypher statements for particular questions:
{examples}

The question is:
{question}"""

class QueryProcessor:
    def __init__(self):
        self.groq_api_key = None
        self.neo4j_url = None
        self.neo4j_user = None
        self.neo4j_password = None
        self.query_chain = None
        self.refinement_chain = None
        self.selected_cypher_llm = LLAMA3
        self.selected_cypher_model_provider = "Groq" 
        self.selected_qa_llm = LLAMA3
    
    def initialize(self):
        # Load keys and initialize all required objects
        self._initialize_keys()
        self.query_chain = self._get_query_chain()
        self.refinement_chain = self._get_refinement_chain()

    def _initialize_keys(self):
        # Initialize keys only once
        load_dotenv()
        self.groq_api_key = os.getenv('GROQ_API_KEY') or st.secrets.get("GROQ_API_KEY")
        self.neo4j_url = os.getenv('NEO4J_URL') or st.secrets.get("NEO4J_URL")
        self.neo4j_user = os.getenv('NEO4J_USER') or st.secrets.get("NEO4J_USER")
        self.neo4j_password = os.getenv('NEO4J_PASSWORD') or st.secrets.get("NEO4J_PASSWORD")
        print("Keys initialized")

    def set_cypher_llm_choice(self, provider, llm):
        self.selected_cypher_llm = llm
        self.selected_cypher_model_provider = provider
        self.query_chain = self._get_query_chain()

    def set_qa_llm_choice(self, llm_choice):
        self.selected_qa_llm = llm_choice
        self.query_chain = self._get_query_chain()

    def _get_cypher_llm(self):
        if(self.selected_cypher_model_provider == "Groq"):
            return ChatGroq(temperature=0, model_name=self.selected_cypher_llm)
        if(self.selected_cypher_model_provider == "Ollama"):
            return ChatOllama(temperature=0, model=self.selected_cypher_llm)

    def _get_qa_llm(self):
        return ChatGroq(temperature=0, model_name=self.selected_qa_llm)

    def _get_prompt(self):
        return PromptTemplate(
            input_variables=["schema", "question", "examples"], template=CYPHER_GENERATION_TEMPLATE
        )

    def _get_graph(self):
        graph = Neo4jGraph(url=self.neo4j_url, username=self.neo4j_user, password=self.neo4j_password, refresh_schema=True)
        print(graph.schema)
        return graph

    def _get_query_chain(self):
        # Create a chain that combines the LLMs and graph
        return GraphCypherQAChain.from_llm(
            allow_dangerous_requests=True,
            cypher_llm=self._get_cypher_llm(),
            qa_llm=self._get_qa_llm(),
            graph=self._get_graph(),
            verbose=True,
            cypher_prompt=self._get_prompt(),
            return_intermediate_steps=True,
            validate_cypher=True,
        )

    REFINEMENT_TEMPLATE="""
    You are an expert in prompt refinement and java software development.
    You task is to improve the origin user question/prompt by providing an alternative prompt, that is more concise and expressive.

    The user question will later be used to generate a cypher query for a knowledge graph.
    The knowledge graph is created from parsed java code. So the entities will be s.th. like class, method, field, package, etc.
    The relations will be s.th. like 
    * a class declares methods
    * a package contains classes/interfaces/enums
    * a method has parameter
    * a method has a return type
    * a class implements an interface

    If you don't find a better alternative return the origin user question.
    Do only return the alternative prompt, no additional text.

    As an additional input you get the schema of the graph database.
    Please use the names of the entities, properties and relationships of the schema, to make sure that the improved prompt is
    close the underlying schema, in order to simplify the subsequent cyper generation.
    {schema}

    The origin user question is:
    {question}
    """

    def _get_refinement_llm(self):
        return ChatGroq(temperature=0, model_name=LLAMA3)

    def _get_refinement_chain(self):
        prompt = PromptTemplate.from_template(template=self.REFINEMENT_TEMPLATE)
        chain = prompt | self._get_refinement_llm() | StrOutputParser()
        return chain

    def handle_query(self, query: str):
        # Handle a single query and return results
        result = self.query_chain.invoke({'query': query, 'examples' : QUERY_EXAMPLES})
        steps = result['intermediate_steps']

        # Extract Cypher query, result, and final answer
        cypher_query = steps[0]['query']
        cypher_result = pd.DataFrame(steps[1]['context'])
        final_answer = result['result']

        return cypher_query, cypher_result, final_answer
