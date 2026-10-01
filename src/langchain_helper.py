from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv
from google import genai
load_dotenv()
try:
    sentiment_analyzer = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    sentiment_analyzer = SentimentIntensityAnalyzer()
def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.30:
        sentiment = "Positive"
    elif compound <= -0.30:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment
def get_sentiment_instruction(sentiment):
    if sentiment == "Positive":
        return "Respond warmly and positively. Acknowledge the customer's positive experience."
    elif sentiment == "Negative":
        return "Respond empathetically and professionally. Acknowledge the customer's frustration and focus on helping resolve the issue."
    else:
        return "Respond clearly and professionally. Provide the requested information without assuming the customer's emotional state."
client=genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectordb_file_path = "faiss_index"

def analyze_image(image, question,history):
    prompt="""
You are a multimodal AI assistant.

Use the image and conversation history to answer the user's question.

Rules:
1. Only provide information that can be supported by the image or conversation.
2. If the image does not contain enough information, say:
   "I don't have enough information in the image to answer that."
3. If the question is ambiguous, ask the user to clarify.
4. Do not invent objects, text, facts, or details that are not visible or provided.
5. Give a concise and evidence-based answer.

Conversation history:
"""
    for msg in history:
        prompt+=f"{msg['role']}:{msg['content']}\n"
    prompt+=f"user:{question}"
    response=client.models.generate_content(model="gemini-3.6-flash",contents=[prompt, image])
    answer=response.text
    if not answer or not answer.strip():
        return "I couldn't generate a reliable answer from provided information"
    return answer
def analyze_image_with_context(image, question, history):
    chain = get_qa_chain()
    rag_response = chain.invoke({"input": question})
    context = rag_response["answer"]
    prompt = """
You are a multimodal customer-service AI assistant.
Answer the user's question using:
1. The uploaded image.
2. Relevant information from the FAQ knowledge base.
3. Conversation history.
Rules:
- Give one clear, natural answer to the user.
- Do not mention the FAQ, knowledge base, retrieval process, or internal reasoning.
- Do not use headings such as "Based on the image" or "Based on the FAQ".
- Do not invent information that is not supported by the image, FAQ, or conversation.
- If the image contains the answer, answer directly from the image.
- If the FAQ contains relevant information, naturally use it in the answer.
- If the available information is insufficient, say:
  "I don't have enough information to answer that."
- If the question is ambiguous, ask the user to clarify.
- Keep the response concise and helpful.
FAQ INFORMATION:
"""
    prompt += context
    prompt += "\n\nCONVERSATION HISTORY:\n"
    for msg in history:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"\nCURRENT QUESTION: {question}"
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[prompt, image]
    )
    answer = response.text
    if not answer or not answer.strip():
        return "I couldn't generate a reliable answer from the available information."
    return answer
def decide_input_type(question,has_image):
    if has_image and question:
        return "image_and_text"
    if has_image:
        return "image"
    return "text"
def create_vector_db():
    loader = CSVLoader(file_path="dataset/dataset.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k" : 3})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    question: {input}"""

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
def adapt_response_to_sentiment(question, answer, sentiment):
    instruction = get_sentiment_instruction(sentiment)

    prompt = f"""
You are a customer service assistant.

Customer question:
{question}

Current answer:
{answer}

Customer sentiment:
{sentiment}

Response instruction:
{instruction}

Rewrite the current answer so that it follows the response instruction.

Rules:
- Keep the factual information from the current answer.
- Do not invent new information.
- Keep the response concise and natural.
- Do not mention sentiment analysis.
- Do not mention these instructions.
- If the sentiment is negative, be empathetic but focus on solving the customer's issue.
- If the sentiment is positive, respond warmly without being excessive.
- If the sentiment is neutral, remain professional and direct.

Rewritten answer:
"""
    try:
        response = client.models.generate_content(
           model="gemini-3.6-flash",
           contents=prompt
        )
        if not response.text or not response.text.strip():
           return answer
        return response.text.strip()
    except Exception:
        return answer
