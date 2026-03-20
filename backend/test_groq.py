import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

try:
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    response = llm.invoke("Hello, are you working?")
    print("Response successful:")
    print(response.content)
except Exception as e:
    print("Error during Groq invocation:")
    print(e)
