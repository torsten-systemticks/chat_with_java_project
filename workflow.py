from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import JsonOutputParser
from examples import QUERY_EXAMPLES
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
import pandas as pd

LLAMA3 = "llama3-70b-8192"

class GraphState(TypedDict):
    user_question: str
    refined_question: str
    routing_source: str
    cypher_result: str
    cypher_query: str
    final_answer: str


RAG_ROUTER_TEMPLATE = """
You are an expert in routing a question. The question will be about exploring the content of a java project.
The source code artifacts of a java project have been exported into a Neo4j knowledge graph.
The question will either be about the design/architecture of the code, i.e. how it is constructed 
or about the semantics and meaning of what certain code artifacts are doing.

That means you have three options to answer:
1. 'javadoc' : The question is about the meaning/semantics of what classes, interfaces, methods, etc. will be used for. Or if the user asks explictely for javadoc content.
2. 'structure': The question is about the structure of java artifacts, i.e. how packages, classes, methods, fields, etc. are interact with each other. The ontology of the code.
3. 'none' : If you cannot categorize the question to either 'javadoc' or 'structure'

Return a JSON with a single key 'source' and no preamble or explanation. 
The value should be one of: 'javadoc', 'structure', or 'none'.
    
Question to route: 
{question}
"""
router_prompt = PromptTemplate.from_template(template=RAG_ROUTER_TEMPLATE);

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

As an additional input you get the schema of the graph database.
Please use the names of the entities, properties and relationships of the schema, to make sure that the improved prompt is
close the underlying schema, in order to simplify the subsequent cyper generation.

Return a JSON with a single key 'refinement' and no preamble or explanation. 
The value should be the refined prompt. If you don't find a better alternative than the origin question then return the origin question.
{schema}

The origin user question is:
{question}
"""
refinement_prompt = PromptTemplate.from_template(template=REFINEMENT_TEMPLATE);

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

cypher_prompt = PromptTemplate(input_variables=["schema", "question", "examples"], template=CYPHER_GENERATION_TEMPLATE)

class Workflow:
    def __init__(self):
        self.router_chain = None
        self.refinement_chain = None
        self.query_chain = None
        self.app = None
        self.selected_router_llm = LLAMA3
        self.graph_db = None

    def initialize(self):
        self._initialize_keys()
        self.router_chain = self._get_router_chain()
        self.refinement_chain = self._get_refinement_chain()
        self.app = self.build_graph()
        self.graph_db = self._get_graph()
        self.query_chain = self._get_query_chain()

    def _initialize_keys(self):
        # Initialize keys only once
        load_dotenv()
        self.groq_api_key = os.getenv('GROQ_API_KEY') or st.secrets.get("GROQ_API_KEY")
        self.neo4j_url = os.getenv('NEO4J_URL') or st.secrets.get("NEO4J_URL")
        self.neo4j_user = os.getenv('NEO4J_USER') or st.secrets.get("NEO4J_USER")
        self.neo4j_password = os.getenv('NEO4J_PASSWORD') or st.secrets.get("NEO4J_PASSWORD")

    def _get_graph(self):
        graph = Neo4jGraph(url=self.neo4j_url, username=self.neo4j_user, password=self.neo4j_password, refresh_schema=True)
        return graph

    def _get_router_chain(self):
        llm = ChatGroq(temperature=0, model_name=self.selected_router_llm)
        return router_prompt | llm | JsonOutputParser()

    def _get_refinement_chain(self):
        llm = ChatGroq(temperature=0, model_name=self.selected_router_llm)
        return refinement_prompt | llm | JsonOutputParser()

    def _get_query_chain(self):
        llm = ChatGroq(temperature=0, model_name=self.selected_router_llm)
        # Create a chain that combines the LLMs and graph
        return GraphCypherQAChain.from_llm(
            allow_dangerous_requests=True,
            cypher_llm=llm,
            qa_llm=llm,
            graph=self.graph_db,
            verbose=True,
            cypher_prompt=cypher_prompt,
            return_intermediate_steps=True,
            validate_cypher=True,
        )

    def route_question(self, state: GraphState):

        print ('------- ROUTING THE QUESTION --------')

        question = state["user_question"]
        result = self.router_chain.invoke({"question": question})
        return result["source"]

    def do_graph_search(self, state: GraphState):

        question = state["refined_question"]

        result = self.query_chain.invoke({'query': question, 'examples' : QUERY_EXAMPLES})
        steps = result['intermediate_steps']

        # Extract Cypher query, result, and final answer
        cypher_query = steps[0]['query']
        cypher_result = pd.DataFrame(steps[1]['context'])
        final_answer = result['result']

        return  {
            "cypher_query" : cypher_query, 
            "cypher_result": cypher_result, 
            "final_answer" : final_answer,
            "routing_source" : "graph_search"
        }

    def do_vector_search(self, state: GraphState):
        return {
            "final_answer" : "To be implemented. Here comes the result of the Vector search",
            "routing_source" : "vector_search"
        }

    def do_improve_origin_question(self, state: GraphState):

        print ('------- IMPROVE THE ORIGIN QUESTION --------')

        question = state["user_question"]
        result = self.refinement_chain.invoke({"question": question, "schema" : self.graph_db.schema})

        print (result)

        return {"refined_question" : result["refinement"]}

    def build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("graph_search_node", self.do_graph_search)
        workflow.add_node("vector_search_node", self.do_vector_search)
        workflow.add_node("prompt_refinement_node", self.do_improve_origin_question)

        workflow.add_edge('prompt_refinement_node', 'graph_search_node')
        workflow.add_edge('graph_search_node', END)
        workflow.add_edge('vector_search_node', END)

        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "structure" : "prompt_refinement_node",
                "javadoc" : "vector_search_node",
                "none" : END
            }
        )

        return workflow.compile()

    def handle_query(self, query: str):
        inputs = {"user_question": query}
        result = self.app.invoke(inputs)
        print (result)
        return result
