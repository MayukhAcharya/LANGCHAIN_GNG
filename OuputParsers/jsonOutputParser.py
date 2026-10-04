# get json output of the llm
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model='google/gemma-4-31B-it')

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(
    template="give me 5 facts about {topic}.\n {format}",
    input_variables=["topic"],
    partial_variables={'format': parser.get_format_instructions()}
)

chain=template | model | parser

result=chain.invoke({
    'topic':"wormhole"
})

print(result)