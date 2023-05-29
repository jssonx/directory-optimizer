################################################################################
# FILE: ../LangBase\src\main.py
################################################################################

import os
import tempfile
import streamlit as st
from streamlit_chat import message
from utils.langbase import Langbase

st.set_page_config(page_title="langbase", page_icon=":book:", layout="wide")


def initialize_session_state():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
        if is_openai_api_key_set():
            st.session_state["langbase"] = Langbase(st.session_state["OPENAI_API_KEY"])
        else:
            st.session_state["langbase"] = None


def update_openai_api_key():
    if (
        len(st.session_state["input_OPENAI_API_KEY"]) > 0
        and st.session_state["input_OPENAI_API_KEY"]
        != st.session_state["OPENAI_API_KEY"]
    ):
        st.session_state["OPENAI_API_KEY"] = st.session_state["input_OPENAI_API_KEY"]
        if st.session_state["langbase"] is not None:
            st.warning("Please upload files again.")
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""
        st.session_state["langbase"] = Langbase(st.session_state["OPENAI_API_KEY"])


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if (
        st.session_state["user_input"]
        and len(st.session_state["user_input"].strip()) > 0
    ):
        query = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            answers_from_gpt = st.session_state["langbase"].ask(query)

        st.session_state["messages"].append((query, True))
        st.session_state["messages"].append((answers_from_gpt, False))


def read_and_save_file():
    st.session_state["langbase"].forget()  # to reset the knowledge base
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(
            f"Ingesting {file.name}"
        ):
            st.session_state["langbase"].ingest(file_path)
        os.remove(file_path)


def is_openai_api_key_set() -> bool:
    return len(st.session_state["OPENAI_API_KEY"]) > 0


def get_token_nums():
    converted_messages = convert_messages(st.session_state["messages"])
    return st.session_state["langbase"].get_tokens(converted_messages)


def convert_messages(raw_messages):
    messages = []
    for msg, is_user in raw_messages:
        messages.append({"role": "user" if is_user else "assistant", "content": msg})
    return messages


def display_session_state():
    st.subheader("Session State")
    st.write("messages:", st.session_state["messages"])
    st.write("langbase:", st.session_state["langbase"])
    st.write("token_nums:", get_token_nums())

def display_titles():
    st.write(
            """
            ✨ langbase &nbsp; [![GitHub][github_badge]][github_link]
            =====================

            A LangChain-based knowledgebase QA dock.

            [github_badge]: https://badgen.net/badge/icon/GitHub?icon=github&color=black&label
            [github_link]: https://github.com/jssonx/langbase
            """
        )
    

def main():
    display_titles()
    initialize_session_state()


    if st.text_input(
        "OpenAI API Key",
        value=st.session_state["OPENAI_API_KEY"],
        key="input_OPENAI_API_KEY",
        type="password",
    ):
        update_openai_api_key()

    st.subheader("Upload files")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
        disabled=not is_openai_api_key_set(),
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input(
        "Start here",
        key="user_input",
        disabled=not is_openai_api_key_set(),
        on_change=process_input,
    )

    st.divider()
    display_session_state()  # 添加显示会话状态的部分


if __name__ == "__main__":
    main()

# Questions:
# Who is the author of the paper "Attention is all you need"?
# What is MACK CROLANGUAGE's email address?


################################################################################
# FILE: ../LangBase\src\utils\cal_tokens.py
################################################################################

import tiktoken

def get_num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return get_num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return get_num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

################################################################################
# FILE: ../LangBase\src\utils\langbase.py
################################################################################

import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from pprint import pprint
from utils.cal_tokens import get_num_tokens_from_messages

from supabase import Client, create_client
from langchain.vectorstores import SupabaseVectorStore


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Langbase:
    def __init__(self, openai_api_key=None, vectorstore_type="chroma") -> None:
        self.vectorstore_type = vectorstore_type
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.supabase_client = Client(supabase_url, supabase_key)
        self.embeddings = OpenAIEmbeddings(client=None, openai_api_key=openai_api_key)
        self.persist_directory = "./data/vectorstore"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="human_input"
        )
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"],
            template="""You are a great expert.
                context: {context}
                chat_history: {chat_history}
                Human: {human_input}
                Expert:""",
        )
        self.llm = ChatOpenAI(
            client=None,
            model_name="gpt-3.5-turbo",
            temperature=0.9,
            openai_api_key=openai_api_key,
        )
        if self.vectorstore_type == "chroma":
            if self.vectorstore_exists_chroma() == True:
                self.db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                ).as_retriever()
                self.chain = load_qa_chain(
                    self.llm, chain_type="stuff", memory=self.memory, prompt=self.prompt
                )
                pprint("Vectorstore exists.")
            else:
                self.db = None
                self.chain = None
        elif self.vectorstore_type == "supabase":
            if self.vectorstore_exists_supabase() == True:
                self.db = SupabaseVectorStore(
                    embedding=self.embeddings,
                    client=self.supabase_client,
                    table_name="documents",
                ).as_retriever()
                self.chain = load_qa_chain(
                    self.llm, chain_type="stuff", memory=self.memory, prompt=self.prompt
                )
                pprint("Vectorstore exists.")
            else:
                self.db = None
                self.chain = None

    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        elif self.db is None:
            response = "Please, ingest a document first."
        else:
            docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, human_input=question)
        return response

    def ingest(self, file_path: str) -> None:
        pdf_loader = PyMuPDFLoader(file_path)
        loaded_documents = pdf_loader.load()
        split_documents = self.text_splitter.split_documents(loaded_documents)
        if self.vectorstore_type == "chroma":
            self.db = Chroma.from_documents(
                documents=split_documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            ).as_retriever()
        elif self.vectorstore_type == "supabase":
            self.db = SupabaseVectorStore.from_documents(
                documents=split_documents,
                embedding=self.embeddings,
                client=self.supabase,
                table_name="documents",
            ).as_retriever()
        self.chain = load_qa_chain(
            self.llm, chain_type="stuff", memory=self.memory, prompt=self.prompt
        )

    def get_tokens(self, messages) -> int:
        return get_num_tokens_from_messages(messages)

    def vectorstore_exists_chroma(self) -> bool:
        if os.path.exists(self.persist_directory):
            return len(os.listdir(self.persist_directory)) > 0
        return False

    def vectorstore_exists_supabase(self) -> bool:
        response = self.supabase_client.table("documents").select("id").execute()
        return len(response.data) > 0

    def forget(self) -> None:
        self.db = None
        self.chain = None

################################################################################
# FILE: ../LangBase\src\utils\supabase_utils.py
################################################################################

import os
from dotenv import load_dotenv
import supabase

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_api_key = os.getenv("SUPABASE_SERVICE_KEY")

client = supabase.Client(supabase_url, supabase_api_key)

def get_row_count():
    response = client.table('documents').select('id').execute()
    return len(response.data)

