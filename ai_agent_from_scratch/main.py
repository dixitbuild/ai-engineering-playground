# Tutorial: How to create an AI Agent from scratch using LangChain and LLMs (OpenAI, Anthropic, Groq)
# Youtube: https://www.youtube.com/watch?v=bTMPwUgLZf0

#load env var
from dotenv import load_dotenv

# help with structured output and validation of the output from LLMs
from pydantic import BaseModel

#langchain LLMs support
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
#tool related imports
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor #bunch of tools that can be used to create an agent

#prompt template - helps to create a prompt for the LLMs. so we can give more information to what we want the LLM to do. This is useful for more complex tasks.
from langchain_core.prompts import ChatPromptTemplate
# for stuctured output/models
from langchain_core.output_parsers import PydanticOutputParser

# import the tools that we have created in tools.py
from tools import search_tool, wiki_tool
#help in debugging.
from debug_help import DebugLLMCallback

load_dotenv()

# That how we use OpenAI, Anthropic, and Groq LLMs in LangChain. You can choose the model that best fits your needs.
llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0.3, max_tokens=2000)
# llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=1000)
# llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0, max_tokens=1000)

#step 1.
#Using the LLM to invoke a simple prompt, but not added agent functionality yet. This is just a simple example of how to use the LLMs in LangChain.
# response = llm_groq.invoke("whats the meaning of life?")
# print('--->', response)

#step 2.
# we will define a simple python class which will specify the type of content that we want our LLM to generate. This is useful for more complex tasks where we want to have more control over the output of the LLM.
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

#here we are telling in prompt that we want the output to be in a specific format.
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools you need to atleast check one site before answering. 
            Wrap the output in this format and provide no other text and Provide a summary in at most 80 words only \n{format_instructions} \n """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ('placeholder', '{agent_scratchpad}')]).partial(format_instructions=parser.get_format_instructions()) # also asked in promot what resuponse should be in a specific format.

print('parser.get_format_instructions()---->', parser.get_format_instructions())
# print('Prompt created---->', prompt)

tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What can i help you research? ")
# raw_response = agent_executor.invoke({"query": query or "What is the meaning of life?"})
# print('raw_response---->', raw_response)

raw_response = agent_executor.invoke(
    {"query": query or "What is the meaning of life?"},
    config={"callbacks": [DebugLLMCallback()]}
)

print("FINAL OUTPUT:", repr(raw_response.get("output")))


try:
    structured_response = parser.parse(raw_response.get('output'))
except Exception as e:
    print('**************************************')
    print('Error parsing response:', e)
    print()
    print('raw_response---->', raw_response)
    structured_response = None

print('Topic:---->', structured_response.topic if structured_response else None)
print('Summary:---->', structured_response.summary if structured_response else None)
print('Sources:---->', structured_response.sources if structured_response else None) 
print('Tools Used:---->', structured_response.tools_used if structured_response else None)   
# print('structured_response---->', structured_response)