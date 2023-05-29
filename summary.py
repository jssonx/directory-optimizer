################################################################################
# FILE: ../LangBase\src\main.py
################################################################################

from fastapi import FastAPI
from src.scripts.cors import setup_cors
from pydantic import BaseModel
from src.scripts.load_env import OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN
from src.scripts.embeddings import create_embeddings, load_embeddings
from src.scripts.models import models_openai
from pprint import pprint

app = FastAPI()
setup_cors(app)
chain = models_openai()
db = load_embeddings()

@app.post("/ask-gpt/v1")
def answer_questions_with_GPT(queries):
    prompt = "This is an html page form, you need to fill in my resume with the content of the form to give me the information I should fill in the form. If you can't answer based on existing information, just answer none."
    query = prompt + " : " + queries
    docs = db.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    # pprint(element)
    return answer

@app.get("/test")
def test():
    return {"msg": "Server is running!"}

# uvicorn src.main:app --reload

################################################################################
# FILE: ../LangBase\src\scripts\cors.py
################################################################################

from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头部
    )

################################################################################
# FILE: ../LangBase\src\scripts\embeddings.py
################################################################################

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from src.scripts.pdf_processor import process_pdf

def create_embeddings(pdf_path):
    texts = process_pdf(pdf_path)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    # use save_local to save the db to a local file
    db.save_local('./data/vectorstore/resume_index')
    return db

def load_embeddings():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("./data/vectorstore/resume_index", embeddings)
    return db


################################################################################
# FILE: ../LangBase\src\scripts\load_env.py
################################################################################

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

################################################################################
# FILE: ../LangBase\src\scripts\models.py
################################################################################

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

def models_openai():
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

################################################################################
# FILE: ../LangBase\src\scripts\pdf_processor.py
################################################################################

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


def process_pdf(file_path):
    # Handle the local knowledge base
    reader = PdfReader(file_path)

    # Read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    return texts

