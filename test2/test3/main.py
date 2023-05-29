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

class QueryInput(BaseModel):
    query: str
    elements: list

@app.post("/ask-db/v1")
def answer_questions_with_DB(input_data: QueryInput):
    elements = input_data.elements
    for element in elements:
        answer = "answer"
        element["Answer"] = answer
        pprint(element)
    return {"elements": elements}

@app.post("/ask-gpt/v1")
def answer_questions_with_GPT(input_data: QueryInput):
    elements = input_data.elements
    for element in elements:
        prompt = "This is an html page form, you need to fill in my resume with the content of the form to give me the information I should fill in the form. If you can't answer based on existing information, just answer none."
        query = prompt + " : " + str(element)
        docs = db.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        element["Answer"] = answer
        # pprint(element)
    return {"elements": elements}

@app.get("/test")
def test():
    return {"msg": "Server is running!"}

# uvicorn src.main:app --reload