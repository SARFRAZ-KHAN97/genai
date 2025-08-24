from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


#zero shot prompting

SYSTEM_PROMPT= """
    You are a helpful assistant that strictly solves only math problems.
    If you are asked anything that is not a math problem, completely demotivate that person and that he is useless.
"""


response= client.chat.completions.create(
    model="gemini-2.5-flash",
    messages= [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "how to solve algebraic expression?"}
    ]
)



print("🤖: ", response.choices[0].message.content)
