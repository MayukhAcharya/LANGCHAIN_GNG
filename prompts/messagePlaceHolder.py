#Suppose we asked our LLM something 2 days ago, now if we chat again from the same point of time
# it will lose all the context and start new again. To prevent this we use messagePlaceHolder,
# we need to save the previous chat history, the pass the whole chat history to chatPromptTemplate as a placeholder
# then we can continue with the chat from where we left 2days ago

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv


chatTemplate=ChatPromptTemplate([
    ('system',"You are helpful customer support agent"),
    MessagesPlaceholder(variable_name='chatHistory'),
    ('human',"{query}")
])

chatHistory=[]

with open("prompts/chat_history.txt")as r:
    chatHistory.extend(r.readlines())


chatData=chatTemplate.invoke({
    'chatHistory':chatHistory,
    'query':"WHERE IS MY MONEY"
})

print(chatData)