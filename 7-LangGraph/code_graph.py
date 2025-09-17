from typing_extensions import TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import Literal


load_dotenv()


client= OpenAI(
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
)


class ClassifyMsgResponse(BaseModel):
    isCoding_question: bool



class CodeAccuracyResponse(BaseModel):
    accuracy_percentage: str



class State(TypedDict):
    user_query: str
    llm_result: str | None
    accuracy_percentage: str | None
    isCoding_question: bool | None



def classify_msg(state: State):
    print("💭")
    query= state["user_query"]

    SYSTEM_PROMPT= """
        You are an AI assistant. Your job is to detect if the user's query is
        related to coding question or not.
        Return the response in specified JSON boolean only.
    """

    #Structured response using pydantic(for now available only in beta for openai & parse need to be used)
    response= client.beta.chat.completions.parse(
        model= "gemini-1.5-flash",
        response_format= ClassifyMsgResponse,
        messages= [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )

    isCoding_question= response.choices[0].message.parsed.isCoding_question
    state["isCoding_question"]= isCoding_question

    return state




#we tell beforehand the possible routes(nodes) 
def route_query(state: State) -> Literal["general_query", "coding_query"]:
    print("🔀")
    is_coding= state["isCoding_question"]

    if is_coding:
        return "coding_query"
    else:
        return "general_query"




def general_query(state: State):
    print("😊")
    query= state["user_query"]

    response= client.chat.completions.create(
        model= "gemini-2.5-flash",
        messages= [
            {"role": "user", "content": query}
        ]
    )
    state["llm_result"]= response.choices[0].message.content

    return state



def coding_query(state: State):
    print("💻")
    query= state["user_query"]

    SYSTEM_PROMPT= """
        You are an expert coding assistant. Your job is to help the user with
        coding related questions. Provide code snippets if necessary.
    """

    response= client.chat.completions.create(
        model= "gemini-2.5-pro",
        messages= [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )
    state["llm_result"]= response.choices[0].message.content

    return state




def coding_validate_query(state: State):
    print("✅")
    query= state["user_query"]
    llm_result= state["llm_result"]

    SYSTEM_PROMPT= f"""
        You are an expert in calculating accuracy percentage of the code
        according to the user's query.
        Return the percentage accuracy in specified JSON format only.

        User's query: {query}
        Code provided by the AI: {llm_result}
    """

    response= client.beta.chat.completions.parse(
        model= "gemini-2.5-flash",
        response_format= CodeAccuracyResponse,
        messages= [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Calculate the accuracy percentage of the code according to the user's query."}
        ]
    )

    state["accuracy_percentage"]= response.choices[0].message.parsed.accuracy_percentage

    return state




graph_builder= StateGraph(State)

#define nodes
graph_builder.add_node("classify_msg", classify_msg)
graph_builder.add_node("route_query", route_query)
graph_builder.add_node("general_query", general_query)  
graph_builder.add_node("coding_query", coding_query)
graph_builder.add_node("coding_validate_query", coding_validate_query)

#define edges
graph_builder.add_edge(START, "classify_msg")
graph_builder.add_conditional_edges("classify_msg", route_query) #takes routing as a function

graph_builder.add_edge("general_query", END)
graph_builder.add_edge("coding_query", "coding_validate_query")
graph_builder.add_edge("coding_validate_query", END)

graph= graph_builder.compile()





def main():
    user= input("👤: ")

    #invoke the graph
    initial_state= {
        "user_query": user,
        "llm_result": None,
        "accuracy_percentage": None,
        "isCoding_question": False
    }

    #graph_result= graph.invoke(initial_state)
    #print("graph result:", graph_result)

    for event in graph.stream(initial_state):
        print("🤖: ", event)


main()


