import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from google import genai
from lingua import Language, LanguageDetectorBuilder
import argostranslate.translate
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    sentiment_analyzer = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    sentiment_analyzer = SentimentIntensityAnalyzer()

load_dotenv()
language_detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.HINDI,
    Language.SPANISH,
    Language.FRENCH
).build()
def detect_language(text):
    detected_language = language_detector.detect_language_of(text)
    if detected_language == Language.HINDI:
        return "Hindi"
    elif detected_language == Language.SPANISH:
        return "Spanish"
    elif detected_language == Language.FRENCH:
        return "French"
    elif detected_language == Language.ENGLISH:
        return "English"
    else:
        return "Unknown"

def detect_languages(text):
    detected_segments = language_detector.detect_multiple_languages_of(text)
    language_codes = {
        Language.ENGLISH: "English",
        Language.HINDI: "Hindi",
        Language.SPANISH: "Spanish",
        Language.FRENCH: "French"
    }
    return [
        {
            "start": segment.start_index,
            "end": segment.end_index,
            "language": language_codes.get(segment.language, "Unknown")
        }
        for segment in detected_segments
    ]
def translate_mixed_to_english(text):
    detected_segments = language_detector.detect_multiple_languages_of(text)
    language_codes = {
        Language.ENGLISH: "English",
        Language.HINDI: "Hindi",
        Language.SPANISH: "Spanish",
        Language.FRENCH: "French"
    }
    translated_parts = []
    for segment in detected_segments:
        segment_text = text[segment.start_index:segment.end_index]
        language = language_codes.get(segment.language, "Unknown")

        if language == "English" or language == "Unknown":
            translated_parts.append(segment_text)
        else:
            translated_parts.append(
                translate_to_english(segment_text, language)
            )
    return " ".join(translated_parts)
def translate_to_english(text, language):
    language_codes = {
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "English": "en"
    }
    source_code = language_codes.get(language)
    if not source_code or source_code == "en":
        return text
    return argostranslate.translate.translate(
        text,
        source_code,
        "en"
    )

def translate_from_english(text, language):
    language_codes = {
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "English": "en"
    }
    target_code = language_codes.get(language)
    if not target_code or target_code == "en":
        return text
    return argostranslate.translate.translate(
        text,
        "en",
        target_code
    )
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

def get_qa_chain(history=""):
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k" : 3})

    prompt_template = """Given the following conversation history, context, and question, generate an answer based on the context only.
       Use the conversation history to understand references, follow-up questions, and the user's intent.
       In the answer try to provide as much text as possible from the "response" section in the source document context without making much changes.
       If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
       Rules:
       - Answer the user's question directly and naturally.
       - Use the conversation history only to understand the meaning of the current question.
       - Do not mention the conversation history, context, or retrieval process.
       - Do not say "According to the context", "Based on the context", "According to the provided information", or similar phrases.
       - Do not explain your reasoning.
       - Preserve the factual information from the response section.
       - If the answer is not found in the context, say "I don't know."

       CONVERSATION HISTORY:
       {history}

       CONTEXT:
       {context}

       QUESTION:
       {input}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "input", "history"]
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

