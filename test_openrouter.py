import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

try:
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku"),
        messages=[
            {"role": "user", "content": "Hello !! Which model are you ??"},
        ],
        temperature=1.5
    )

    print("✅ API WORKING")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ API FAILED")
    print(e)