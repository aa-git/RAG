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

pdf_path = "./modern_history_spectrum.pdf"

loader = PDFMinerLoader(pdf_path)
pdf_content = loader.load()

#print(type(pdf_content), pdf_content[0])


CHUNK_SIZE = 1000
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


out = embeddings.embed_documents(texts = ['embed this oooo'])





