import os

from dotenv import load_dotenv

from google import genai


# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


# Configure Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==========================================
# Generate Gemini Response
# ==========================================

def generate_gemini_response(
    query,
    context
):

    prompt = f"""
You are NeuralWiki AI Assistant.

You MUST answer ONLY using
the provided context.

Rules:
- Do not hallucinate.
- If answer is not in context,
  say:
  "The uploaded documents do
   not contain that information."
- Be concise but accurate.
- Mention technologies,
  entities, and concepts clearly.

Context:
{context}

Question:
{query}
"""

    try:

        response = (
            client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
        )

        return response.text

    except Exception as error:

        print(
            "Gemini Error:",
            error
        )

        return (
            "Gemini API error occurred."
        )