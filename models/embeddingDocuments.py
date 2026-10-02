from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv() #load the env

llm=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-mpnet-base-v2" )

query=["Hello, this is a new query","qyery","Bold"]

embedData=llm.embed_documents(query)
print(str(embedData))