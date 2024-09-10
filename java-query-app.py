import streamlit as st
import pandas as pd

import query_transformer as qt

from dotenv import load_dotenv

load_dotenv()

chain = qt.get_query_chain()

def handle_query(user_query):
 
    result = qt.invoke_with_query(user_query, chain)
    steps = result['intermediate_steps']

    cypher_query = steps[0]['query']
    cypher_result = pd.DataFrame(steps[1]['context'])
    final_answer = result['result']

    return cypher_query, cypher_result, final_answer

# Function to display the Cypher query and result
def display_results(cypher_query, cypher_result, final_answer):
    st.subheader("Generated Cypher Query")
    st.code(cypher_query, language='cypher')
    
    st.subheader("Query Result")
    st.table(cypher_result)

    st.subheader("Final Answer")
    st.write(final_answer)

# Streamlit UI
st.title("Inspect your Java Project")

user_query = st.text_area("Enter your query in natural language:", height=100)

if st.button("Submit"):
    if user_query:
        cypher_query, cypher_result, final_answer = handle_query(user_query)
        display_results(cypher_query, cypher_result, final_answer)
    else:
        st.error("Please enter a query.")


# Run the app with: streamlit run your_script_name.py
