from dotenv import load_dotenv
from langfuse.openai import OpenAI
import os

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """You are a helpful assistant that will help the user with their queries. who solves user query within 40 words. You will be given a user query and you will respond with a helpful answer. You will not ask the user for more information. You will not provide any additional information that is not relevant to the user's query. You will not provide any personal opinions or advice"""

def main():
    while True:
        user_input = input("User: Enter your message: > ")
        if user_input.lower() in ['quit', 'exit']:
            break
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )
        print("Assistant:", response.choices[0].message.content)

main()