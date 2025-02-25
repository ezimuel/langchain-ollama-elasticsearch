# Store PDF documents in Elasticsearch using Ollama embeddings
# by Enrico Zimuel (enrico@zimuel.it)

import glob, os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

# Check if the index already exists
res = vector_db.client.indices.exists(index=index_name)
if res.body:
    print(f"The index {index_name} already exists in Elasticseach")
    exit(1)
    
print(f"Reading the PDFs in {base_path}/data")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
# Read the PDF files and split into chunks
all_splits = []
for file in glob.glob(f"{base_path}/data/*.pdf"):
    loader = PyPDFLoader(file)
    docs = loader.load()
    pages=len(docs)
    print(f"Read {file} with {pages} pages")
    chunks = text_splitter.split_documents(docs)
    num_chunks=len(chunks)
    print(f"Splitted in {num_chunks} chunks")
    all_splits.append(chunks)
            
print(f"Storing chunks in Elasticsearch")
for chunks in all_splits:
    # Index the chunks to Elasticsearch
    vector_db.from_documents(
        documents=chunks,
        embedding=embeddings,
        es_connection=vector_db.client,
        index_name=index_name
    )
    chunks_size=len(chunks)
    print(f"Stored {chunks_size} chunks in {index_name} index")

