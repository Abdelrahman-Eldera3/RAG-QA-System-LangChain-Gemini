# EdTech Q&A System using RAG & Google Gemini 🧠🌱

## 📌 Project Overview
An intelligent Question & Answering system designed to provide accurate responses based on a specific knowledge base (CSV). It uses **RAG (Retrieval-Augmented Generation)** to ensure the LLM (Google Gemini) provides factual answers grounded in the provided context, avoiding hallucinations.

## 🛠️ Technical Stack
* **LLM:** Google Gemini Pro (via `langchain-google-genai`).
* **Framework:** LangChain.
* **Vector Database:** FAISS (Facebook AI Similarity Search).
* **Embeddings:** HuggingFace Embeddings.
* **UI:** Streamlit.
* **Data Handling:** CSVLoader (LangChain).

## 🏗️ Architecture & Features
* **Vector Store Creation:** Converts FAQ data into vector embeddings and stores them locally using FAISS for fast similarity search.
* **RAG Pipeline:** When a user asks a question, the system retrieves the most relevant context from the vector database and passes it to Gemini Pro.
* **Custom Prompting:** Uses a specialized `PromptTemplate` to ensure the model stays within the context and answers "I don't know" if the information is missing.
* **User Interface:** A clean Streamlit web app for real-time interaction.

## 🚀 How to Run
1. Clone the repository.
2. Create a `.env` file and add your Google API Key:
   `GOOGLE_API_KEY=your_actual_key_here`
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run the Streamlit app:
   `streamlit run main.py`

## 📂 Project Structure
* `main.py`: Streamlit frontend logic.
* `Langchain_helper.py`: Backend logic for RAG, vector DB, and chain creation.
* `codebasics_faqs.csv`: The knowledge base (source data).

## 👨‍💻 Author
**Abdelrahman Elderaa** - AI Engineer
* **LinkedIn:** [Abdelrahman Elderaa](http://linkedin.com/in/abdelrahman-elderaa-405b7b244)