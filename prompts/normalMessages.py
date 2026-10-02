from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model="google/gemma-4-31B-it")

model=ChatHuggingFace(llm=llm)

#in here there is no chat-history, AI does not know the context of chat
# while True:
#     userMsg=input("You: ")
#     if userMsg=="exit":
#         break
#     result=model.invoke(userMsg)
#     print(f"AI: {result.content}")


#adding a chat hiistory
chatHistory=[]

while True:
    userMsg=input("You: ")
    chatHistory.append(userMsg)
    if userMsg=="exit":
        break
    result=model.invoke(chatHistory)
    chatHistory.append(result.content)
    print(f"AI: {result.content}")