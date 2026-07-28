import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title(" CUSTOMER SERVICE CHATBOT 🤖")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()
    st.success("KnowledgeBase Created")
question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain.invoke({"input":question})

    st.header("Answer")
    st.write(response["answer"])