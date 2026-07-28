from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# # Initialize instructor embeddings using the Hugging Face model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectordb_file_path = "faiss_index"


def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path="dataset/dataset.csv", source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(search_kwargs={"k" : 3})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {input}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "input"]
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=PROMPT,
    )
    chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("hello?"))