from fastapi import FastAPI
import uvicorn
from LLMCore import generation_llama_3_2, vector_store_in_memory

app = FastAPI()

@app.get("/server-health-check")
def server_health_check():
    return {"running": "True"}

@app.get("/query/{query}")
def query(query: str):
    from LLMPlayGround import wikipedia_extract
    content = wikipedia_extract.get_content()
    vector_store = vector_store_in_memory.put_in_qdrant_vector_store(content)
    answer = generation_llama_3_2.generate_response_to_query(query, vector_store)
    return {"answer to query '"+query+"'": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.1.5", port=8000)
