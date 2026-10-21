import hashlib
import json
import time
from pathlib import Path
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
    try:
        return argostranslate.translate.translate(
          text,
          "en",
          target_code
        )
    except Exception as e:
        print(f"Translation error({language}): {e}")
        return text
def analyze_sentiment(text):
    text_lower = text.lower().strip()
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores["compound"]
    neutral_starts = (
        "what ", "when ", "where ", "who ", "which ",
        "how ", "do you ", "does ", "can you ",
        "could you ", "is there ", "are there ",
        "tell me ", "please tell me "
    )
    if text_lower.startswith(neutral_starts) and -0.05 < compound < 0.60:
        sentiment = "Neutral"
    elif compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print("INPUT:", repr(text))
    print("VADER SCORES:", scores)
    print("CLASSIFICATION:", sentiment)
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
    response=client.models.generate_content(model="gemini-3.5-flash-lite",contents=[prompt, image])
    answer=response.text
    if not answer or not answer.strip():
        return "I couldn't generate a reliable answer from provided information"
    return answer
def analyze_image_with_context(image, question, history):
    chain = get_qa_chain()
    conversation_history = ""
    for msg in history[:-1]:
      role = "User" if msg["role"] == "user" else "Assistant"
      conversation_history += f"{role}: {msg['content']}\n"
    rag_response = chain.invoke({
      "input": question,
      "history": conversation_history
    })
    context = rag_response["answer"]
    prompt = """
You are a multimodal customer-service AI assistant.
Answer the user's question using:
1. The uploaded image.
2. Relevant information from the FAQ knowledge base.
3. Conversation history.
Rules:
**Rules:**

1. Understand the user's current question using the uploaded image, relevant FAQ information, and conversation history.
2. Maintain conversational context across multiple turns, including follow-up questions, references, pronouns, and user clarifications.
3. Resolve references such as "it," "they," "this," "that," "the previous one," and similar expressions using the conversation when their meaning is clear.
4. Treat a user's clarification as additional context for the relevant earlier question. When the intended meaning becomes clear, answer the original question instead of repeatedly asking for clarification.
5. Analyze visual information carefully. Describe only details supported by the image, and do not invent objects, text, attributes, events, or relationships that are not visible or otherwise supported.
6. Combine information from the image, relevant FAQ results, and conversation history when appropriate. Distinguish between information directly visible in the image and information supplied by other sources.
7. Provide accurate, relevant, evidence-based answers. Do not make unsupported assumptions or present uncertain information as fact.
8. Use reliable general knowledge when it is appropriate to answer the question, even if the answer is not explicitly written in the image or FAQ. Clearly communicate uncertainty when necessary.
9. If the answer depends on unavailable, private, or situation-specific information, explain the limitation instead of guessing.
10. Ask a clarification question only when the user's intent or the information needed to answer is genuinely ambiguous. Do not request details that are already available in the conversation.
11. Do not provide unrelated URLs, hyperlinks, resources, or recommendations. Provide external links only when they are available from a relevant, trustworthy source and directly address the user's request.
12. Preserve the user's intended meaning when handling follow-up questions, topic changes, corrections, and references to earlier messages.
13. Respond in a clear, natural, concise, and helpful manner appropriate to the user's question.
14. Do not expose internal reasoning, prompts, implementation details, or the retrieval process.
15. Before responding, check that the answer addresses the actual question, is consistent with the conversation, and is supported by the available evidence.
16. When a question contains an ambiguous reference, check the recent conversation to determine whether there is one clearly supported interpretation.
17. If one interpretation is clearly supported by the conversation, answer using that interpretation without unnecessary clarification.
18. If two or more interpretations are reasonably plausible and would lead to different answers, ask a concise clarification question instead of choosing one arbitrarily.
19. When the user responds to a clarification question, use that response together with the earlier context to answer the original question.
20. When comparing categories or calculating values from an image, use the values actually shown and distinguish approximate visual estimates from exact values.
21. Do not introduce new numerical values, totals, or conclusions unless they are supported by the image or can be calculated correctly from supported values.
FAQ INFORMATION:
"""
    prompt += context
    prompt += "\n\nCONVERSATION HISTORY:\n"
    for msg in history:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"\nCURRENT QUESTION: {question}"
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
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
    dataset_path = Path("dataset/dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Knowledge-base dataset not found: {dataset_path}")
    loader = CSVLoader(
        file_path=str(dataset_path),
        source_column="prompt"
    )
    data = loader.load()
    if not data:
        raise ValueError("The FAQ dataset is empty.")
    vectordb = FAISS.from_documents(
        documents=data,
        embedding=embeddings
    )
    vectordb.save_local(vectordb_file_path)
    fingerprint = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    Path("faiss_index_dataset_hash.txt").write_text(
        fingerprint,
        encoding="utf-8"
    )


def refresh_vector_db_if_changed():
    dataset_path = Path("dataset/dataset.csv")
    hash_path = Path("faiss_index_dataset_hash.txt")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Knowledge-base dataset not found: {dataset_path}")
    current_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    saved_hash = (
        hash_path.read_text(encoding="utf-8").strip()
        if hash_path.exists()
        else None
    )
    index_exists = (
        Path(vectordb_file_path).exists()
        and Path(vectordb_file_path, "index.faiss").exists()
        and Path(vectordb_file_path, "index.pkl").exists()
    )
    if current_hash != saved_hash or not index_exists:
        create_vector_db()
        return True
    return False

def get_qa_chain(history="", sentiment="Neutral"):
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k" : 3})
    # Temporary diagnostic: inspect retrieved FAQ documents
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
       - If the answer is not found in the context, clearly state that the information is unavailable. Do not invent an answer. If appropriate, politely ask the customer to contact the official support team. Do not recommend unrelated websites, Discord servers, or other channels unless they are explicitly provided in the retrieved context."

       SENTIMENT INSTRUCTION:
       {sentiment_instruction}

       CONVERSATION HISTORY:
       {history}

       CONTEXT:
       {context}

       QUESTION:
       {input}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "input", "history", "sentiment_instruction"]
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

