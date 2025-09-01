from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import json
import requests
import os

load_dotenv()


client= OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def get_weather(city: str):
    url= f"https://wttr.in/{city}?format=%C+%t"
    response= requests.get(url)

    if response.status_code == 200:
        return f"The current weather in {city} is: {response.text}."
    
    return "Sorry, I couldn't fetch the weather information right now."




def run_command(cmd: str):
    result= os.system(cmd)
    return result




available_tools= {
    "get_weather": get_weather,
    "run_command": run_command
}




#If someone asks you for the current date and time, respond with the current date and time in the format "YYYY-MM-DD HH:MM:SS".
#You can access current date and time using {datetime.now()}
SYSTEM_PROMPT= f"""
    You are a helpful assistant.
    You work on start, plan, action, observe mode.

    For the given user query and available tools, plan the step by step execution.
    Based on the planning, select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.

    Rules:
    - Follow Output JSON format strictly
    - Always perform one step at a time and wait for the next input.
    - Carefully read the user query and understand it before planning the action.

    Output format:
    {{
        "step": "string", 
        "content": "string",
        "function": "Name of the function to be called or 'None', when step is action",
        "input": "The input to the function or 'None'"
    }}
    
    Available Tools:
    - "get_weather(city: str)": Takes a city name as input and returns the current weather in that city.
    - "run_command(cmd: str)": Takes a shell command as input and executes it on the system.


    Example:
    Input: What is the weather in New York City?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}

"""


messages= [
    {"role": "system", "content": SYSTEM_PROMPT}
]



while True:
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

        if parsed_response.get("step") == "plan":
            print("🧠: ", parsed_response.get("content"))
            continue

        
        if parsed_response.get("step") == "action":
            tool_name= parsed_response.get("function")
            tool_input= parsed_response.get("input")

            print(f"🛠️: Calling tool: {tool_name} with input: {tool_input}")

            if tool_name in available_tools:
                output= available_tools[tool_name](tool_input)
                messages.append({"role": "user", "content": json.dumps({"step": "observe", "output": output})})
                continue

        
        if parsed_response.get("step") == "output":
            print("🤖: ", parsed_response.get("content"))
            break