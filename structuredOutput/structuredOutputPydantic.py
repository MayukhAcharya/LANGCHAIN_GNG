from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from pydantic import BaseModel,Field
from typing import Annotated,Literal
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model='google/gemma-4-31B-it',task="text-generation")

model=ChatHuggingFace(llm=llm)

class Review(BaseModel):
    keyThemes:list[str]=Field(description="Write down all the key themes discussed in the review")
    reviewSummary:str=Field(description='Give a small 5 line summary of the review')
    pros:list[str]=Field(description="Write down all the pros of the item given in the review")
    cons:list[str]=Field(description="Write down all the cons of the item given in the review")
    sentiment:Literal["positive","negative"]=Field(description="What is the sentiment of the review?")

structuredModel=model.with_structured_output(Review,include_raw=True,method="json_mode")

result=structuredModel.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

print(result['raw'])