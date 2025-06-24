from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
import os
from langchain_core.documents import Document
from uuid import uuid4

####  Controlling Variables
RECONSTRUCT_VECTOR_STORE = True

import wikipedia_extract

content = wikipedia_extract.get_content()

documents = []
ids = []
for piece in content:
    documents += [Document(page_content=piece, metadata={"source" : "txt file"})]
    ids += [str(uuid4())]
    
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


embeddings = OllamaEmbeddings(
    model="llama3.2",
)

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)

vector_store.add_documents(
    documents=documents,
    ids = ids
)

print("done: doc inserted in qdrant(in memory)")


'''
for res in results:
    print(res.page_content, res.metadata)
'''

### generation part
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model='llama3.2')
chain = load_qa_chain(llm, chain_type="stuff")


while True:
    query = input("\n\n\n\n\n\n\nquery: ")
    results = vector_store.similarity_search(query, k=4)
    
    answer = chain.run(input_documents = results, question=query)
    print(answer)