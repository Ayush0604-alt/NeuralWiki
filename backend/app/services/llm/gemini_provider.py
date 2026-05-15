import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

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
"""


def generate_gemini_response(query, context):

    prompt = f"""{SYSTEM_PROMPT}

Context from retrieved documents:
{context}

User question:
{query}
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )
        return response.text

    except Exception as error:
        print("Gemini Error:", error)
        return "**Error:** Could not reach the Gemini API. Please check your API key and try again."