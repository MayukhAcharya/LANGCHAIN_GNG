# in structured output parser, there is no validation of the output, To validate it we will use pydantic
# why use:
# 1. Strict Schema enforcement 2. Type Safety 3. Easy Validation 4. Seamless Integration

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model='google/gemma-4-31B-it',temperature=1)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name:Optional[str]=Field(default=None,description="Name of the peson")
    age:Optional[int]=Field(gt=18,description="Age of the person",default=None)
    city:Optional[str]=Field(description="City of the person where he/she is living",default="")
    errorMessage:str=Field(description="If wrong/invalid country passed add error message")


parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="""Give me name, age, city of a Person living in {country}, 
    if passed a country name which is wrong/invalid then return an error Message  \n {format}""",
    partial_variables={"format":parser.get_format_instructions()},
    input_variables=["country"]
)

chain=template | model | parser

result=chain.invoke({
    "country":"Chile"
})

print(result)

resultDict=result.model_dump()
print(resultDict)