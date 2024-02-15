import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def translate_statement(statement, target_language):
    # Translate to the target language
    prompt = f"Please provide the {target_language} translation for these sentences: {statement}."
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    translated_statement = response.choices[0].message.content

    return translated_statement


input_file_path = "dataset/comments copy.csv"
output_file_path = "dataset/fix_jp.csv"

data = pd.read_csv(input_file_path, header=None)

# Perform translations
all_translations = []
for index, row in data.iterrows():
    statement = row[0]  
    translation = translate_statement(statement, "Japanese")
    
    retranslated_back_to_english = translate_statement(translation, "English")
    
    all_translations.append({
        "statement": statement,
        "japanese": translation,
        "retranslate": retranslated_back_to_english
    })

translations_df = pd.DataFrame(all_translations)
translations_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")

print("Translations are done")
