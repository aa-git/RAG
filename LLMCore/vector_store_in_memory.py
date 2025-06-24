from langchain_core.documents import Document
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def put_in_qdrant_vector_store(content):
    '''
    content is a list of strings, these string are the content
    '''
    documents = []
    ids = []
    for piece in content:
        documents += [Document(page_content=piece, metadata={"NA" : "NA"})]
        ids += [str(uuid4())]
    


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

    return vector_store