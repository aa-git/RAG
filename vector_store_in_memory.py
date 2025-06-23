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

results = vector_store.similarity_search("Tell me about earth", k=2)

for res in results:
    print(res.page_content, res.metadata)





while True:
    input()


if RECONSTRUCT_VECTOR_STORE: 
    Qdrant.from_documents()
    qdrant = Qdrant.from_documents(
        docs, 
        embeddings,
        path = "./tmp/local_qdrant",
        collection_name="my_documents",
        force_recreate=True
    )
else:
    Qdrant.construct_instance(path="./tmp/local_qdrant", collection_name="my_documents")

print("exiting"
      )
import sys
sys.exit()

print("finding relevant docs from qdrant, for given query")
query = "was partition of bengal cancelled ?"
found_docs = qdrant.similarity_search(query)

print("count of relevant docs found = > ",len(found_docs))
for doc_ in found_docs:
    print(doc_.page_content)
    print("------------------------------------------")
    print("------------------------------------------")



### generation part
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model='llama3.2')

chain = load_qa_chain(llm, chain_type="stuff")
found_docs_2 = qdrant.similarity_search(query)


answer = chain.run(input_documents = found_docs_2, question=query)

print(answer)

