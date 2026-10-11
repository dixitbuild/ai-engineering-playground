#load env var
from dotenv import load_dotenv

from pydantic import BaseModel

#langchain LLMs support
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

#prompt template - helps to create a prompt for the LLMs. so we can give more information to what we want the LLM to do. This is useful for more complex tasks.
from langchain_core.prompts import ChatPromptTemplate
# for stuctured output/models
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# That how we use OpenAI, Anthropic, and Groq LLMs in LangChain. You can choose the model that best fits your needs.
# llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=1000)
# llm_anthropic = ChatAnthropic(model_name="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=1000)
llm_groq = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7, max_tokens=1000)

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
    ("system", "You are a research assistant that will help genterate a research paper."
    "Answer the user query and use neccessary tools."
    "Wrap the output in this formate and provide no other tesn \n{format_instructions}"),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ('placeholder', '{agent_scratch_tools}')]).partial(format_instructions=parser.get_format_instructions())