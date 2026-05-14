import os

from dotenv import load_dotenv

from openai import OpenAI


load_dotenv()

NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY"
)


client = OpenAI(

    api_key=NVIDIA_API_KEY,

    base_url="https://integrate.api.nvidia.com/v1"
)


# ==========================================
# Generate NVIDIA Response
# ==========================================

def generate_nvidia_response(
    query,
    context
):

    prompt = f"""
You are NeuralWiki AI Assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{query}
"""

    try:

        completion = client.chat.completions.create(

            model="meta/llama-3.1-70b-instruct",

            messages=[

                {
                    "role": "system",

                    "content":
                        "You are a helpful AI assistant."
                },

                {
                    "role": "user",

                    "content": prompt
                }
            ],

            temperature=0.3,

            max_tokens=1024
        )

        return (
            completion
            .choices[0]
            .message
            .content
        )

    except Exception as error:

        print(
            "NVIDIA ERROR:",
            error
        )

        return (
            "NVIDIA API error occurred."
        )