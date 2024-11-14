import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

client = AzureOpenAI (
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    api_version="2024-02-15-preview",
    azure_endpoint="https://recitation8.openai.azure.com/"
)

def get_language(post: str) -> str:
    context = f"What language is this in:\n\n{post}? Return just the language. If you cannot determine the language, return 'Cannot determine language'."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": context
            }
        ]
    )
    return response.choices[0].message.content
    

def get_translation(post: str) -> str:
    context = f"Translate the following text to English:\n\n{post}. Return just the translation. If the text is has no meaningful phrases or is just numbers, return 'Unintelligible or malformed text' as the translation"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": context
            }
        ]
    )
    return response.choices[0].message.content


def translate_content(content: str) -> tuple[bool, str]:
  
    if not isinstance(content, str):
        return (False, "Content is not a string")
    
    if (content == ""):
        return (False, "Content is an empty string")

    language = get_language(content)

    if language == "Cannot determine language":
        return (False, "Cannot determine language")

    if language == "English":
        return (True, content)

    translated_text = get_translation(content)

    if translated_text == "Unintelligible or malformed text":
        return (False, "Translation failed")
    else:
        return (False, translated_text)
    

