from fastapi import FastAPI

app = FastAPI()

@app.get("/server-health-check")
def server_health_check():
    return {"running": "True"}

@app.get("/query/{query}")
def query(query: str):
    pass

