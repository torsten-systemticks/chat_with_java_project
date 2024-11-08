import streamlit as st 
import pandas as pd
from workflow import Workflow

# Initialize Query Processor once per session
if 'query_processor' not in st.session_state:
    st.session_state['query_processor'] = Workflow()
    st.session_state['query_processor'].initialize()

st.sidebar.title("Configuration")

cypher_ll_options = [
     ("Groq", "llama3-70b-8192"),
     ("Groq",  "mixtral-8x7b-32768"), 
     ("Groq", "gemma2-9b-it"),
     ("Groq", "llama3-groq-70b-8192-tool-use-preview"),
     ("Ollama", "codegemma:instruct")    
]

formatted_options = [f'{provider}: {model}' for provider, model in cypher_ll_options]

cypher_llm_choice = st.sidebar.selectbox('Select a Language Model:', formatted_options, index=0)


qa_llm_choice = st.sidebar.selectbox(
    "Choose the QA Model:",
    ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it",
     "llama3-groq-70b-8192-tool-use-preview"],  # Dropdown options for LLMs
    index=0  # Default to the first LLM
)

provider, llm = cypher_llm_choice.split(': ')

#st.session_state['query_processor'].set_cypher_llm_choice(provider, llm)
#st.session_state['query_processor'].set_qa_llm_choice(qa_llm_choice)

# Streamlit UI
st.title("Inspect your Java Project")

user_query = st.text_area("Enter your query in natural language:", height=100)

# Process and display results when the submit button is clicked
if st.button("Submit"):
    if user_query:
        result = st.session_state['query_processor'].handle_query(user_query)

        if 'refined_question' in result:
            st.subheader('Automated refined question')
            st.write(result['refined_question'])

        if 'cypher_query' in result:
            st.subheader("Generated Cypher Query")
            st.code(result['cypher_query'], language='cypher')

        if 'cypher_result' in result:
            st.subheader("Query Result")
            st.table(result['cypher_result'])

        st.subheader("Final Answer")
        st.write(result['final_answer'])
    else:
        st.error("Please enter a query.")
