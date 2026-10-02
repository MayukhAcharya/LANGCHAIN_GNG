from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query="Who is known as God of Cricket?"
llm=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-mpnet-base-v2" )


#embed the document
embedDocuments=llm.embed_documents(documents)

#embed the query
embedQuery=llm.embed_query(query)

# in cosine similarity both params should be 2D list. embedDocuments is already a 2D list
scores=cosine_similarity([embedQuery],embedDocuments)[0]

value=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(f"Query: {query}")
print(f"Confidence score: {abs(value[1])*100}")
print(documents[value[0]])


#This is a small example of a RAG system