import os
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title(" CUSTOMER SERVICE CHATBOT 🤖")

upload_file=st.file_uploader("Upload FAQ CSV", type=["csv"])

if upload_file:
    with open("dataset/dataset.csv","wb") as f:
        f.write(upload_file.getbuffer())

    st.success("FAQ CSV uploaded successfully")

btn=st.button("Create KnowledgeBase")

if btn:
    with st.spinner("Creating KnowledgeBase. Please wait"):
        create_vector_db()
    st.success("KnowledgeBase Created successfully. You can ask questions now")

question = st.text_input("Question: ")

if question:
    if not os.path.exists("faiss_index"):
        st.error("Please create KnowledgeBase first")
    else:
        chain = get_qa_chain()
        response = chain.invoke({"input":question})

        st.header("Answer")
        st.write(response["answer"])