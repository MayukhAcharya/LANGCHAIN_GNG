#static messages with chat history

# There are 3 types of Message in Langchain for prompt :
# 1. System Message -> Message that is passed at the very beginning of chat, a system message
# 2. Human Message -> Messages/prompt by humans/us
# 3. AI Message -> Messages returned by AI

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model="google/gemma-4-31B-it")

model=ChatHuggingFace(llm=llm)

#langchain chat history
messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="tell me about langchain")
]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)