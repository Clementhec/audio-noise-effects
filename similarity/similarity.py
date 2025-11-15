import pandas as pd
import numpy as np
from ast import literal_eval
from utils.embeddings_utils import get_embedding, cosine_similarity


datafile_path = "data/soundbible_details_from_section_with_embeddings.csv"
df = pd.read_csv(datafile_path)

df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

def vector_similarity(df, product_description, n=20, pprint=True):
    product_embedding = get_embedding(
        product_description,
        model="text-embedding-3-small"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


results = search_reviews(df, "La plupart des îles sont montagneuses, parfois volcaniques.", n=3)
print(results)
