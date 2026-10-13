from dotenv import load_dotenv
from typing import Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model('gpt-4', temperature=0.0, max_tokens=2000)

class State(TypedDict):
    """State of the graph."""
    messages: Annotated[list, add_messages]
    # message will be a list, whenever we need to change messages we will use add_messages function to add messages to the list. This will be used to keep track of the messages in the graph.
    # fist we add system message, then user message while we invoke the graph, then we add assistant message(llm will give this) to the list. This will be used to keep track of the messages in the graph.

graph_builder = StateGraph(State)


def chatbot(state: State) -> State:
    """Chatbot function that takes in a state and returns a state."""
    messages = state['messages']
    response = llm.invoke(messages)
    messages.append(response)
    return {'messages': messages}

#node
graph_builder.add_node('chatbot', chatbot)

#edge 
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

user_input = input("User: Enter your message: ")
state = graph.invoke({'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': user_input}]})

print('------>',state['messages'], len(state['messages']))
print(state['messages'][-1].content)