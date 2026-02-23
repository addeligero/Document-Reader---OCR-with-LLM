from groq import Groq
from dotenv import load_dotenv
import os, json

load_dotenv()

def call_llm(prompt: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    if not os.getenv("GROQ_API_KEY"):
        return {"error": "GROQ_API_KEY not found in environment"}

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You must respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        print("RAW:", raw)

        return json.loads(raw)

    except Exception as e:
        return {
            "error": str(e),
            "raw_output": response.choices[0].message.content if "response" in locals() else None
        }


if __name__ == "__main__":
    result = call_llm('why groq is the best?')
    print(result)