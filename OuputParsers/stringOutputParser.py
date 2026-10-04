# strOutputParser returns the output as string
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model='google/gemma-4-31B-it')

model=ChatHuggingFace(llm=llm)

#prompt 1
template1=PromptTemplate(template="Give me a detailed summary on {topic}",input_variables=["topic"])

#prompt 2
template2=PromptTemplate(template="Give me a 5 line summary of the following, {text}",input_variables=["text"])

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser
# template1 send to model, then model will get the output, then parser will clean the data then repeat the process for template2

result=chain.invoke({'topic':'womhole'})

print(result)