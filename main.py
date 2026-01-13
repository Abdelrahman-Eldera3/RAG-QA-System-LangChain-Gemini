import streamlit as st
from Langchain_helper import Create_vector_DB,get_QA_Chain

st.title("Codebasics Q&A 🌱")
bnt=st.button("Create Knowledge")
if bnt:
    Create_vector_DB()

question = st.text_input("Question: ")

if question:
    chain=get_QA_Chain()
    response=chain.invoke(question)

    st.header("Answer")
    st.write(response)