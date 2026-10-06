from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Spam Email Detection Backend is running"}