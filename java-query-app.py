import streamlit as st 
import pandas as pd
from query_transformer import QueryProcessor

# Initialize Query Processor once per session
if 'query_processor' not in st.session_state:
    st.session_state['query_processor'] = QueryProcessor()
    st.session_state['query_processor'].initialize()

st.sidebar.title("Configuration")
cypher_llm_choice = st.sidebar.selectbox(
    "Choose the Query Generation Model:",
    ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it",
     "llama3-groq-70b-8192-tool-use-preview"],  # Dropdown options for LLMs
    index=0  # Default to the first LLM
)

qa_llm_choice = st.sidebar.selectbox(
    "Choose the QA Model:",
    ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it",
     "llama3-groq-70b-8192-tool-use-preview"],  # Dropdown options for LLMs
    index=0  # Default to the first LLM
)

st.session_state['query_processor'].set_cypher_llm_choice(cypher_llm_choice)
st.session_state['query_processor'].set_qa_llm_choice(qa_llm_choice)

# Streamlit UI
st.title("Inspect your Java Project")

user_query = st.text_area("Enter your query in natural language:", height=100)

# Process and display results when the submit button is clicked
if st.button("Submit"):
    if user_query:
        cypher_query, cypher_result, final_answer = st.session_state['query_processor'].handle_query(user_query)
        st.subheader("Generated Cypher Query")
        st.code(cypher_query, language='cypher')

        st.subheader("Query Result")
        st.table(cypher_result)

        st.subheader("Final Answer")
        st.write(final_answer)
    else:
        st.error("Please enter a query.")
