from dotenv import load_dotenv
from typing import Literal, Annotated
from typing_extensions import TypeDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model('gpt-4', temperature=0.0, max_tokens=2000)

class State(TypeDict):
    """State of the graph."""
    messages: Annotated[list, add_messages]
    # message will be a list, whenever we need to change messages we will use add_messages function to add messages to the list. This will be used to keep track of the messages in the graph.
    # fist we add system message, then user message while we invoke the graph, then we add assistant message(llm will give this) to the list. This will be used to keep track of the messages in the graph.



