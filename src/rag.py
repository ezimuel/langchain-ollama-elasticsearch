# Simple RAG architecture using LangChain, Ollama and Elasticsearch
# by Enrico Zimuel (enrico@zimuel.it)

import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Disable LangChain tracing
#os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Load and chunk contents of the PDF
base_path = Path(__file__).parent.parent.resolve()
  
# Index the chunks in Elasticsearch
dotenv_path = Path(base_path / "elastic-start-local/.env")
if not dotenv_path.is_file():
    print("Error: it seems Elasticsearch has not been installed")
    print("using start-local, please execute the following command:")
    print("curl -fsSL https://elastic.co/start-local | sh")
    exit(1)
    
load_dotenv(dotenv_path=dotenv_path)
index_name="rag-langchain"

# Embeddings
embeddings = OllamaEmbeddings(
    model="llama3.2:3b",
)

vector_db  = ElasticsearchStore(
    es_url=os.getenv('ES_LOCAL_URL'),
    es_api_key= os.getenv('ES_LOCAL_API_KEY'),
    embedding=embeddings,
    index_name=index_name
)

# LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0.0000000001)

# Define prompt for RAG
prompt_template = PromptTemplate.from_template(
    template="Given the following context: {context}, answer to the following question: {question}. Write only three sentences."
)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_db.similarity_search(state["question"], k=8)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.format(question=state["question"], context=docs_content) 
    response = llm.invoke(prompt)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

question = "Who won the Nobel Prize in Physics 2024?"
print(question)
response = graph.invoke({"question": question})
print(response["answer"])