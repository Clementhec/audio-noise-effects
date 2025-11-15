from openai import OpenAI
from pydantic import BaseModel
from typing import List


client = OpenAI()

class HighlightedSentences(BaseModel):
    sentences: List[str]

response = client.responses.parse(
    model="o3-mini",
    input=[
        {"role": "system", "content": "From the given text, highlight 4 exact fragments (≤15 words, no rephrasing) where sound effects would enhance impact."},
        {
            "role": "user",
            "content": "Le Japon est un archipel dont le nombre d'îles varie, suivant les estimations, de 6 852 à 14 125 îles (de plus de 100 m2), dont les quatre plus grandes sont Hokkaidō, Honshū, Shikoku et Kyūshū, représentant à elles seules 95 % de la superficie terrestre du pays. L'archipel s'étend sur plus de trois mille kilomètres. La plupart des îles sont montagneuses, parfois volcaniques. Ainsi, le plus haut sommet du Japon, le mont Fuji (3 776 m), est un volcan dont la dernière éruption a eu lieu en 1707.Le Japon est le onzième pays le plus peuplé du monde, avec un peu moins de 124 millions dhabitants pour 377 975 km2 (328 hab./km2), dont lessentiel est concentré sur les étroites plaines littorales du sud de Honshū et du nord de Shikoku et Kyūshū, formant un ensemble pratiquement urbanisé en continu appelé « Mégalopole japonaise » ou « Taiheiyō Belt » (太平洋ベルト, Taiheiyō beruto?, littéralement « ceinture Pacifique »). Le Grand Tokyo, qui comprend la capitale Tokyo et plusieurs préfectures environnantes, est la plus grande région métropolitaine du monde, avec plus de 35 millions dhabitants. La ville a été première place financière mondiale en 1990.",
        },
    ],
    text_format=HighlightedSentences,
)

event = response.output_parsed
print(event)