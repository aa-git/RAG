RECONSTRUCT_VECTOR_STORE = True
'''
from PyPDF2 import PdfReader

pdf_path = "./modern_history_spectrum.pdf"

reader = PdfReader(pdf_path)

pages = reader.pages

documents = []
for page_ in pages:
    documents.append(page_.extract_text())

print(documents[300])
'''

####
# above example (commented) used pypdf2 to extract 
# data, now we use langchain based python module
# this will take content from pdf, concatenate all pages into 
# one single document, and then split that document to get 
# (almost) equally sized docs
####

#from langchain.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFMinerLoader
#from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
import os

pdf_path = "./history.pdf"

loader = PDFMinerLoader(pdf_path)
pdf_content = loader.load()

#print(type(pdf_content), pdf_content[0])


CHUNK_SIZE = 1000000
CHUNK_OVERLAP = 30

text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
docs = text_splitter.split_documents(pdf_content)

print("------> total docs count created out of pdf file: ",len(docs))

####
# now pdf content has been split into chunk,
# with someoverlap for context, we use openai embedding model
# and qdrant as vector database
# ####



#from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Qdrant   #--old
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings




#embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')

embeddings = OllamaEmbeddings(
    model="llama3.2",
)

print("creating embeddings, and saving in qdrant")




if RECONSTRUCT_VECTOR_STORE: 
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
query = "whT IS 'SAFETY VALVE' THEORY in context of indian history?"
found_docs = qdrant.similarity_search(query)

print("count of relevant docs found = > ",len(found_docs))
for doc_ in found_docs:
    print(doc_.page_content)
    print("------------------------------------------")
    print("------------------------------------------")



### generation part
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2")

chain = load_qa_chain(llm, chain_type="stuff")
found_docs_2 = qdrant.similarity_search(query)


answer = chain.run(input_documents = found_docs_2, question=query)

print(answer)

