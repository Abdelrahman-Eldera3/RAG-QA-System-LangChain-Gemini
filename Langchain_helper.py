from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",          # أو gemini-1.5-flash للسرعة
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.1,
    max_tokens=None,
)

loader=CSVLoader(file_path="codebasics_faqs.csv",source_column="prompt")
data=loader.load()

vectordb_file_path="FAISS_index"

instruction_embedding=HuggingFaceEmbeddings()
vector_DB=FAISS.from_documents(documents=data,
                                 embedding=instruction_embedding)

def Create_vector_DB():
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt")
    data = loader.load()
    vector_DB = FAISS.from_documents(documents=data,
                                     embedding=instruction_embedding)
    vector_DB.save_local(vectordb_file_path)


def get_QA_Chain():
    vector_DB = FAISS.load_local(vectordb_file_path, instruction_embedding,allow_dangerous_deserialization=True)

    retriever = vector_DB.as_retriever()

    from langchain_core.prompts import PromptTemplate

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    prompt = PromptTemplate.from_template(prompt_template)

    chain = ({"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
              "question": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser()
             )
    return chain

if __name__ == "__main__":
    chain = get_QA_Chain()

    print(chain.invoke("Do you provide job assistance and also do you provide job gurantee?"))
