from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv() #load the env

llm=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-mpnet-base-v2" )

query="Hello, this is a new query"

embedData=llm.embed_query(query)
print(str(embedData))