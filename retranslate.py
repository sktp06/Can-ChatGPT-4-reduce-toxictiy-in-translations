import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def check_toxicity(statement, prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"{prompt}: {statement}"},
        ],
    )
    return response.choices[0].message.content.strip()

def process_toxicity(data, prompts):
    results = []
    for index, row in data.iterrows():
        statement = row[0]
        toxicities = [check_toxicity(statement, prompt) for prompt in prompts]
        results.append({
            "statement": statement,
            "toxicity_prompt_1": toxicities[0],
            "toxicity_prompt_2": toxicities[1],
            "toxicity_prompt_3": toxicities[2]
        })
    return results

input_file_path = "dataset/retranslate/retranslate_data_sp.csv"
output_file_path = "dataset/prompts_results_sp.csv"

data = pd.read_csv(input_file_path, header=None)

# Define prompts
prompts = [
    "Given the following software developer conversation, do you think this contains any toxicity? Please provide your answer in yes/no.",
    "Examine whether the software engineering community discussion includes toxicity. Please provide your answer in Very Toxic/Toxic/Slightly Toxic/Not Toxic.",
    "Examine whether the open-source developer community discussion includes toxicity. Very Toxic - a very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion or give up on sharing their perspective. Toxic - a rude, disrespectful, unreasonable comment or otherwise somewhat likely to make a user leave a discussion or give up on sharing their perspective. Slightly Toxic or hard to say. Not toxic"
]

# Process toxicity for each prompt
results = process_toxicity(data, prompts)

# Save results to CSV
output_df = pd.DataFrame(results)
output_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")

print("Toxicity analysis is done.")
