from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv() #load the env

# define what open-source llm model to use and the task to do, in here we are doing task generation
llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",task="text-generation",temperature=0.5,max_new_tokens=50)

#creating the object of chat hugging face
model=ChatHuggingFace(llm=llm)

result=model.invoke("write a poem on the ongoing tech layoffs")

print(result.content)