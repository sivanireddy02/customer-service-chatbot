# Generative AI Q&A: Customer Service Chatbot Using LangChain and Google Gemini

A Generative AI question-answering system that has been developed using LangChain, Streamlit, and a large language model. 
Users can ask questions about courses, internships, and customer service through a simple web interface, after which the chatbot obtains the relevant information from a CSV-based FAQ knowledge base and generates the answers with the aid of a language model.

## Project Highlights

- Answering questions based on frequently asked questions: the knowledge source is a CSV file which contains customer-service questions and answers.
- **Semantic search:** Retrievess relevant FAQ information by using Sentence Transformers and FAISS.
- **Knowledge-base refresh:** The vector index is refreshed whenever the application runs, provided that there have been changes to the FAQ dataset.
- **Sentiment-aware responses:** The system makes use of VADER sentiment analysis in order to adjust its responses according to whether the customer's message is positive, negative, or neutral.
- **Multilingual support:** The text includes facilities for detecting the language and translating it into English, Hindi, Spanish, and French.
- For questions based on images: it makes use of the vision-capable language model that has been configured.
- **Interactive interface:** The interface has been developed using Streamlit so that access to the chatbot is simple.

## Technology Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- Hugging Face Sentence Transformers
- FAISS vector database
- VADER Sentiment Analysis
- Lingua language detection
- Argos Translate
- CSV-based FAQ dataset

## Project Structure

```text
customer-service-chatbot/
├── src/
│   └── langchain_helper.py
│   └── main.py
├── dataset/
│   └── dataset.csv
├── faiss_index/
├── .env
├── requirements.txt
└── README.md
```

The exact files and the directories that are created may differ according to the local setup.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sivanireddy02/customer-service-chatbot.git
   ```

2. Go to the project directory:

   ```bash
   cd customer-service-chatbot
   ```

3. Install the dependencies that are required:

   ```bash
   pip install -r requirements.txt
   ```

4. In the root of the project create a .env file and include the API key variable that is expected by the application:

   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

   Get an API key by using Google's Gemini API setup that is supported. Make sure not to put your actual API key into the repository.

## Usage

1. Make sure the FAQ dataset is in `dataset/dataset.csv`.
2. Start the Streamlit application:

   ```bash
   streamlit run main.py
   ```

3. Open the local URL that is shown in the terminal.
4. Enter a customer-service question and then review the response that has been generated.
Whenever new information becomes available, the FAQ CSV should be updated and the application can then refresh its vector index when it detects changes to the dataset.

## Sample Questions

- Are internships available?
- Do you offer EMI payments?
- Do you offer a JavaScript course?
- Whether I should learn Power BI or Tableau?
- What steps are there for enabling Power Pivot?

The exact answers will differ depending on the information included in the current FAQ dataset.


