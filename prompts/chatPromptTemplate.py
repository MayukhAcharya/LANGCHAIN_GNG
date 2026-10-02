#To create dynamic messages in multi  turn conversations we use chatPromptTemplate 

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


llm=HuggingFaceEndpoint(model="google/gemma-4-31B-it")

model=ChatHuggingFace(llm=llm)

chatTemplate=ChatPromptTemplate([
    ('system',"You are an {domain} expert"),
    ('human',"Explain in simple terms,what is {topic}")
])

chain=chatTemplate | model

result=chain.invoke({
    "domain":"cricket",
    "topic" :"chinaman"
})

print(result.content)