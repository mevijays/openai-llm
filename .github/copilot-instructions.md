# copilot instructions
- use python FASTAPI framework for API and flask for web app.
- use source .venvenv/bin/activate to activate virtual environment.
- use pip install -r requirements.txt to install dependencies.
- use uvicorn main:app --reload to run the API server.
- use flask run to run the web app.
- do not write tests unless asked.
- always use command python3.12 and pip3.12 if python3 or pip3 is mentioned.
- code and dependencies should be compatible with python 3.12.
## Application Structure
- application is a openai like llm server using gpt4all pythonSDK.
- model directory is models/
- model mistral-7b.gguf is main default model.
- all-miniLM-L6-v2 is sentence transformer model for embeddings.
- main.py is the main API server file.
- main.py uses FastAPI framework.
- main.py has /chat ui endpoint for web app.