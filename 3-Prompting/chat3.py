from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()


client= OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


#chain of thought prompting


SYSTEM_PROMPT= """
    You are a helpful assistant that solves user problem by analysing the input and breaking down the problem step by step.

    The steps should be in this given order: 
    1. Analyze the problem
    2. Break down the problem into smaller parts
    3. Solve the smaller parts very precisely
    4. Combine the solutions of the smaller parts to get the final solution
    5. Give the final solution in a concise manner
    6. Always ask a follow up question closely related to the problem to keep the conversation going   
    
    Rules:
    - Always follow the steps in the given order
    - Always perform one step at a time and wait for the next input.
    - Always ask a follow up question closely related to the problem to keep the conversation going
    - Follow JSON format strictly

    Output format:
    {{ "step": "string", "content": "string" }}

    Example:
    Input: explain how a car engine works?
    Output:
    { "step": "Analyze the problem", "content": "User is trying to understand how a car engine works." }
    { "step": "Break down the problem into smaller parts", "content": "Break down the problem into smaller parts." }
    { "step": "Solve the smaller parts very precisely", "content": "Explain the working of each unit ." }
    { "step": "Combine the solutions of the smaller parts to get the final solution", "content": "Give a combined brief description." }
    { "step": "Give the final solution", "content": "Provide the solution in concise, simple and common used language so that user remember it always." }
    { "step": "Ask a follow up question", "content": "What are the different types of car engines? can you tell any?" }
"""



messages= [
    {"role": "system", "content": SYSTEM_PROMPT}
]


query= input("👤: ")
messages.append({"role": "user", "content": query})



while True:
    response= client.chat.completions.create(
        model= "gemini-2.5-flash",
        response_format= {"type": "json_object" },
        messages= messages
    )


    messages.append({"role": "assistant", "content": response.choices[0].message.content})

    parsed_response= json.loads(response.choices[0].message.content)

    if parsed_response.get("step") != "Ask a follow up question":
        print("🤖: ", parsed_response.get("content"))
        continue

    print("🤖: ", parsed_response.get("content"))
    break