from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(model="google/gemma-4-31B-it")

model=ChatHuggingFace(llm=llm)

#The below template is not resuseable, we make it reuseable (reuseableTemplate.py)

# #creating a prompt template
# template=PromptTemplate(template="""
# Please summarize the research paper titled "{paper_input}" with the following specifications:
# Explanation Style: {style_input}  
# Explanation Length: {length_input}  
# 1. Mathematical Details:  
#    - Include relevant mathematical equations if present in the paper.  
#    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
# 2. Analogies:  
#    - Use relatable analogies to simplify complex ideas.  
# If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
# Ensure the summary is clear, accurate, and aligned with the provided style and length.
# """,
# input_variables=["paper_input","style_input","length_input"],
# validate_template=True # this will validate our template inputVariables, if dev forgets to pass it, it will throw error at runtime
# )

#yeh hai aam zindagi
# templateData=template.invoke({
#     'paper_input':"Attention is all you need", # add a research paper
#     'style_input':'simple', # explanation style
#     'length_input':'short' # length of explanation
# })

# result=model.invoke(templateData)

# print(result.content)

template=load_prompt("template.json") #load the template from reuseableTemplate

#yeh hai mentos zindagi
chain=template | model # langchain chaining, first it will go to template it will get the template then it will move to the model 

result=chain.invoke({
    'paper_input':"Attention is all you need", # add a research paper
    'style_input':'simple', # explanation style
    'length_input':'short' # length of explanation
}) # then we need to pass the template data which will be eventually be fed to the chain

print(result.content)