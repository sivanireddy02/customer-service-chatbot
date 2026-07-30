import os
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
import pandas as pd

if "kb_updated" not in st.session_state:
     st.session_state.kb_updated=False
st.title(" CUSTOMER SERVICE CHATBOT 🤖")

upload_file=st.file_uploader("Upload FAQ CSV", type=["csv"],on_change=lambda:st.session_state.update(kb_updated=False))

if upload_file and not st.session_state.kb_updated:
    old_ds=pd.read_csv("dataset/dataset.csv")
    new_ds=pd.read_csv(upload_file)
    merged_ds=pd.concat([old_ds,new_ds],ignore_index=True)
    merged_ds=merged_ds.drop_duplicates(subset=["prompt"])
    merged_ds.to_csv("dataset/dataset.csv",index=False)
    with st.spinner("Updating KnowledgeBase.Please wait"):
            create_vector_db()
    st.success("KnowledgeBase updated successfully")
    st.session_state.kb_updated=True

question = st.text_input("Question: ")

if question:
    if not os.path.exists("faiss_index"):
        st.error("Please create KnowledgeBase first")
    else:
        chain = get_qa_chain()
        response = chain.invoke({"input":question})

        st.header("Answer")
        st.write(response["answer"])