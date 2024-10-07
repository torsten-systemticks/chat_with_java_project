import streamlit as st 
import pandas as pd
from query_transformer import QueryProcessor

# Initialize Query Processor once per session
if 'query_processor' not in st.session_state:
    st.session_state['query_processor'] = QueryProcessor()
    st.session_state['query_processor'].initialize()

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
