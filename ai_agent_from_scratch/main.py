#load env var
from dotenv import load_dotenv

from pydantic import BaseModel

# langchain LLMs support
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

#prompt template
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# That how we use OpenAI, Anthropic, and Groq LLMs in LangChain. You can choose the model that best fits your needs.
# llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=1000)
# llm_anthropic = ChatAnthropic(model_name="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=1000)
llm_groq = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7, max_tokens=1000)

#Using the LLM to invoke a simple prompt, but not added agent functionality yet. This is just a simple example of how to use the LLMs in LangChain.
response = llm_groq.invoke("whats the meaning of life?")
print('--->', response)