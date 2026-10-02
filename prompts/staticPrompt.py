from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model='google/gemma-4-31B-it',task="text-generation",temperature=0.5,max_new_tokens=200)

model=ChatHuggingFace(llm=llm)

result=model.invoke("As an AI how you would you take over the world?")

print(result.content)
