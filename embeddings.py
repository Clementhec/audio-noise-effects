import pandas as pd
import tiktoken

from utils.embeddings_utils import get_embedding

embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000  # the maximum for text-embedding-3-small is 8191

# load & inspect dataset
input_datapath = "data/soundbible_details_from_section.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, sep=";")
df = df[["title", "href", "url", "description", "keywords", "length", "audio_url"]]
print(len(df))
print(df.head(10))

df["embedding"] = df.description.apply(lambda x: get_embedding(x, model="text-embedding-3-small"))
df.to_csv("data/soundbible_details_from_section_with_embeddings.csv")