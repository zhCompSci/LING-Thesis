import os
import pandas as pd
from textgrid import TextGrid

# Your folder path
TEXTGRID_DIR = "."

# # Container for all data
# all_data = []

# # Iterate through all TextGrid files
# for filename in os.listdir(TEXTGRID_DIR):
#     if filename.endswith(".TextGrid"):
#         filepath = os.path.join(TEXTGRID_DIR, filename)
#         tg = TextGrid.fromFile(filepath)

#         # Assume tier 0 contains the transcription info
#         tier = tg[0]

#         # Parse file metadata
#         dialog_id = filename.split("_")[0]
#         speaker = filename.split("_")[1].split(".")[0]

#         # Extract intervals
#         for interval in tier.intervals:
#             if interval.mark.strip():  # ignore empty intervals
#                 all_data.append({
#                     "dialog_id": dialog_id,
#                     "speaker": speaker,
#                     "start_time": interval.minTime,
#                     "end_time": interval.maxTime,
#                     "text": interval.mark.strip()
#                 })

# # Convert to DataFrame and save
# df = pd.DataFrame(all_data)
# df.to_csv("all_textgrid_data.csv", index=False)
# print("Saved to all_textgrid_data.csv")

import pandas as pd
import re

# Load your CSV
df = pd.read_csv("all_textgrid_data3.csv")

# Function to remove "*[[...]]"
def clean_text(text):
    return re.sub(r"\*\[\[.*?\]\]", "", text).strip()

# Apply to text column
df["text"] = df["text"].astype(str).apply(clean_text)

# Save cleaned version
df.to_csv("all_textgrid_data_cleaned.csv", index=False)
print("Saved cleaned CSV to all_textgrid_data_cleaned.csv")

