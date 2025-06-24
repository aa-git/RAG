from langchain.chains.question_answering import load_qa_chain
from langchain_ollama.llms import OllamaLLM

def generate_response_to_query(query, vector_store, results_to_retrieve_from_vector_store = 4):
    llm = OllamaLLM(model='llama3.2')
    chain = load_qa_chain(llm, chain_type="stuff")

    results = vector_store.similarity_search(query, k=results_to_retrieve_from_vector_store)
    
    return chain.run(input_documents = results, question=query) 