from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv
import os
load_dotenv()



class State(TypedDict):
    messages: Annotated[list, add_messages]



llm= init_chat_model(model_provider= "google_genai", model="gemini-2.5-pro", api_key= os.getenv("OPENAI_API_KEY"))


def chat_node(state: State):
    response= llm.invoke(state["messages"])
    return {"messages": [response]}



graph_builder= StateGraph(State)

graph_builder.add_node("chat_node", chat_node)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)



#graph= graph_builder.compile()


def compile_graph_with_checkpointer(checkpointer):
    graph_with_checkpointer= graph_builder.compile(checkpointer= checkpointer)
    return graph_with_checkpointer


def main():
     # Mongo url format:-  mongodb://<username>:<pass>@<host(container_name)>:<port>
    DB_URL= "mongodb://admin:admin@localhost:27017"
    config= { "configurable": {"thread_id":"1"}}  #when thread_id is changed, it will create a new(or reference to existing) state and have access to messages of that thread



    #this creates a fresh new state
    #result= graph.invoke({"messages": [{"role": "user", "content": query}]})
    #and state gets deleted after it, so messages are stored only for all the nodes

     #print("🤖: ", result)
    #print("🤖: ", result["messages"][-1].content)


    with MongoDBSaver.from_conn_string(DB_URL) as mongo_checkpointer:

        graph_with_mongo= compile_graph_with_checkpointer(mongo_checkpointer)

        query= input("👤: ")

        result= graph_with_mongo.invoke({"messages": [{"role": "user", "content": query}]}, config= config)

        print(result)


main()