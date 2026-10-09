import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import streamlit as st
from src.langchain_helper import (get_qa_chain, create_vector_db, analyze_image,analyze_image_with_context, decide_input_type, analyze_sentiment, adapt_response_to_sentiment, detect_language, translate_to_english, translate_mixed_to_english, translate_from_english)
import pandas as pd
from PIL import Image
if "kb_updated" not in st.session_state:
     st.session_state.kb_updated=False
if"messages" not in st.session_state:
     st.session_state.messages=[]
st.title(" CUSTOMER SERVICE CHATBOT 🤖")
upload_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if upload_image:
    image = Image.open(upload_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
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
input_type=decide_input_type(question,upload_image is not None)
if question:
    sentiment = analyze_sentiment(question)
    detected_language = detect_language(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sentiment": sentiment,
        "language" : detected_language
    })
if question:
     input_type=decide_input_type(
          question, upload_image is not None
     )
     if input_type=="image_and_text":
       response = analyze_image_with_context(
          image, question, st.session_state.messages
       )
       response = adapt_response_to_sentiment(
          question, response, sentiment
       )
       st.header("Answer")
       st.write(response)
       st.session_state.messages.append({
         "role": "assistant",
         "content": response
       })
     elif input_type=="image":
       response = analyze_image(
         image, question, st.session_state.messages
       )
       response = adapt_response_to_sentiment(
         question, response, sentiment
       )
       st.header("Answer")
       st.write(response)
       st.session_state.messages.append({
         "role": "assistant",
         "content": response
       })
     else:
          if not os.path.exists("faiss_index"):
            st.error("Please create KnowledgeBase first")
          else:
            english_question = translate_mixed_to_english(question)
            conversation_history = ""
            for message in st.session_state.messages[:-1]:
             message_language = message.get("language", "English")
             message_text = message["content"]

             if message_language != "English":
               message_text = translate_to_english(
                message_text,
                message_language
               )

             role = "User" if message["role"] == "user" else "Assistant"
             conversation_history += f"{role}: {message_text}\n"
            chain = get_qa_chain(conversation_history)

            retrieval_question = (
              conversation_history
              + f"User: {english_question}"
            )

            response = chain.invoke({
               "input": retrieval_question,
               "history": conversation_history
            })
            answer = response["answer"]
            answer = translate_from_english(
               answer,
               detected_language
            )
            answer = adapt_response_to_sentiment(
                question,
                answer,
                sentiment
            )
            st.header("Answer")
            st.write(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
st.subheader("Conversation History")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write("You:", message["content"])
    else:
        st.write("Assistant:", message["content"])