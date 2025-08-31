from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

#few shot prompting

SYSTEM_PROMPT= """
    You are a helpful assistant that solves user problem in a story format.

    Examples: 
    User: explain quantum computing in simple terms
    Assistant: Sure! Imagine you have a magic coin that can be both heads and tails at the same time. Quantum computing is like that magic coin. It uses tiny particles that can be in multiple states at once, allowing computers to solve complex problems much faster than regular computers.    

    You can also use more like real life examples and other relatable real life stories.
"""



response= client.chat.completions.create(
    model= "gemini-2.5-flash",
    messages= [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "explain how aeroplane fly?"}
    ]
)



print("🤖: ", response.choices[0].message.content)