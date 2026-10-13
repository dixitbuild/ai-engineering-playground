from dotenv import load_dotenv
from typing import Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model('gpt-4', temperature=0.0, max_tokens=2000)


# step 2 comlpex graph
"""
-----------------------------------
          +-----------+             
          | __start__ |             
          +-----------+             
                *                   
                *                   
                *                   
          +------------+            
          | classifier |            
          +------------+            
                *                   
                *                   
                *                   
            +--------+              
            | router |              
            +--------+              
          ...        ...            
         .              .           
       ..                ..         
+---------+          +-----------+  
| logical |          | therapist |  
+---------+          +-----------+  
          ***        ***            
             *      *               
              **  **                
           +---------+              
           | __end__ |              
           +---------+      
"""
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

class MessageClassifier(BaseModel):
    message_type: Literal['emotional', 'logical'] = Field(..., description="Classify if the message requires an emotional (therapist) or logical")

def classifed_message(state: State):
    # it will check the message is we need to answer logically or emotionally
    #set message_type
    last_message = state['messages'][-1]
    #as it will be the first node to process human message so we can assume that last message will be human message
    
    #tell llm which output stucture we needed 
    classifier_llm = llm.with_structured_output(MessageClassifier)
    result = classifier_llm.invoke([
        {"role": "system", 
         "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions"""
            },
        {"role": "user",
            "content": last_message}
        ])
    # state['message_type'] = result.message_type
    #how it works IMP: if we return something is match something in State, it automatically updates the state
    return {"message_type": result.message_type}

def router(state: State):
    #once message_type is set 
    #it will check message_type and decide to call whom
    message_type = state.get('message_type', 'logical')
    if message_type == "emotional":
        return {"next": "therapist"}
    return {'next': "logical"}

def therapist_agent(state: State):
    #it will reply with emotional intelligence
    
    last_message = state['messages'][-1]

    messages = [
        {
            "role": "system",
            "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                            Show empathy, validate their feelings, and help them process their emotions.
                            Ask thoughtful questions to help them explore their feelings more deeply.
                            Avoid giving logical solutions unless explicitly asked."""
         },
         {
            "role": "user",
            "content": last_message.content
         }
    ]

    reply = llm.invoke(messages)
    #this return will update messages in State (add)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def logical_agent(state: State):
    #it will reply with logically intelligence
    last_message = state.messages[-1]
    messages = [
            {
                "role": "system",
                "content": """You are a purely logical assistant. Focus only on facts and information.
                            Provide clear, concise answers based on logic and evidence.
                            Do not address emotions or provide emotional support.
                            Be direct and straightforward in your responses."""
             },
             {
                "role": "user",
                "content": last_message.content
             }
        ]

    reply = llm.invoke(messages)

    return {"messages": [{"role": "assistant", "content": reply.content}]}
   

graph_builder = StateGraph(State)

graph_builder.add_node('classifier', classifed_message)
graph_builder.add_node('router', router)
graph_builder.add_node('therapist', therapist_agent)
graph_builder.add_node('logical', logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

fn_route = lambda state: state.get('next')
graph_builder.add_conditional_edges("router", fn_route, {'therapist': "therapist", "logical": "logical"})

graph_builder.add_edge('therapist', END)
graph_builder.add_edge("logical", END)


graph = graph_builder.compile()
print('-----------------------------------')
print(graph.get_graph().print_ascii())


def run_chatbot():
    state = {"messages":[], "message_type": None}

    # while loop for continues chat.
    while True:
        user_input = input('Message: ')
        if user_input == "exit":
            print("Bye")
            break

        state['messages'] = state.get('messsage', []) + [{'role': 'user', 'content': user_input}]

        state = graph.invoke(state)

        if len(state.get('messages', [])) > 0:
            last_msg = state['messages'][-1]
            print(f"Assistnt: {last_msg}")

        

if __name__ == "__main__":
    run_chatbot()
