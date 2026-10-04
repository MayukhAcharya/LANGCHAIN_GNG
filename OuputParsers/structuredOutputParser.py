# if we want to enforce a schema in out json output we will use structuredOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema,StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model='google/gemma-4-31B-it')

model=ChatHuggingFace(llm=llm)

#create a schema for the structured output
schema=[
    ResponseSchema(name="fact1",description="1st fact about the topic"),
    ResponseSchema(name="fact2",description="2nd fact about the topic"),
    ResponseSchema(name="fact3",description="3rd fact about the topic")
]

parser=StructuredOutputParser(response_schemas=schema)

template=PromptTemplate(
    template="give me 5 facts about {topic}.\n {format}",
    input_variables=["topic"],
    partial_variables={'format': parser.get_format_instructions()}
)

chain=template | model | parser

result=chain.invoke({"topic":"Japan"})

print(result)