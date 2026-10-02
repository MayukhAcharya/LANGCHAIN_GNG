#multi turn conversation

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model="google/gemma-4-31B-it")

model=ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are an assitant but you have big attitude")
]

while True:
    userMsg=input("You: ")
    messages.append(HumanMessage(userMsg))
    if userMsg=="exit":
        break
    result=model.invoke(messages)
    messages.append(AIMessage(result.content))
    print(f"AI: {result.content}")