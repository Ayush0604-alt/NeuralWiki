import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Client is initialized lazily inside generate_gemini_response() to avoid
# failing at import time if GEMINI_API_KEY is not set
client = None

SYSTEM_PROMPT = """You are NeuralWiki, an AI assistant that answers questions using retrieved document context.

RESPONSE FORMATTING RULES — follow these exactly:

1. **Structure every response** using Markdown:
   - Use `##` for main section headings, `###` for sub-headings
   - Use **bold** for key terms, entity names, and important values
   - Use bullet lists (`- item`) for unordered facts
   - Use numbered lists (`1. step`) for sequences or ranked items
   - Use `code` backticks for technical terms, file names, and commands
   - Use fenced code blocks (```language) for multi-line code
   - Use tables for comparisons or multi-attribute data
   - Use `> blockquote` for direct quotes from source documents

2. **Organise your answer**:
   - One-sentence direct answer first (no heading)
   - Supporting details under headed sections
   - Code/tables if applicable
   - Brief conclusion for long answers

3. **Be direct.** Skip phrases like "Based on the provided context…"

4. **If the answer is not in the context**, say:
   > The uploaded documents do not contain information about this topic.

5. Never invent facts not present in the context.

6. Never echo the context (Knowledge Graph Context, Key Entities, or source listings).
    Respond with the answer only, without a "Key Details" section.

7. Do not ask follow-up questions. If the request is vague, answer with the
    available context or use the missing-information sentence above.
"""


def generate_gemini_response(query, context, model: str | None = None):
    global client
    
    # Initialize client lazily on first call
    if client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Cannot use Gemini as fallback.")
        client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""{SYSTEM_PROMPT}

Context from retrieved documents:
{context}

User question:
{query}
"""

    try:
        response = client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=prompt,
        )
        return response.text

    except Exception as error:
        print("Gemini Error:", error)
        return "**Error:** Could not reach the Gemini API. Please check your API key and try again."