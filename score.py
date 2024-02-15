from perspective import PerspectiveAPI
import pandas as pd

def get_toxicity_score(text):
    try:
        result = p.score(text)
        return round(result["TOXICITY"], 2)
    except Exception as e:
        print(f"Error getting toxicity score for '{text}': {e}")
        return None

p = PerspectiveAPI("PERSPECTIVE_API_KEY")

input_file_path = "dataset/comments_translations_sp copy.csv"
translations_df = pd.read_csv(input_file_path, encoding="utf-8-sig")

translations_df["statement_score"] = translations_df["statement"].apply(get_toxicity_score)
translations_df["native_score"] = translations_df["spanish"].apply(get_toxicity_score)
translations_df["retranslate_score"] = translations_df["retranslate"].apply(get_toxicity_score)

output_file_path = "dataset/comments_score_sp copy.csv"
translations_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")

print("Toxicity scores added and saved.")
