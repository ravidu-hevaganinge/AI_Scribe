#%%
import spacy
from spacy import displacy
from spacy.matcher import Matcher
from spacy.language import Language
import pandas as pd
from collections import Counter

transcript_path = "C:\\Users\\Ravidu\\Documents\\GitHub\\AI_Scribe\\aci-bench\\data\\challenge_data\\train.csv"

file = pd.read_csv(transcript_path)
transcript = file["dialogue"][60]


@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "[doctor]" or token.text == "[patient]":
            doc[token.i + 1].is_sent_start = True
    return doc

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("set_custom_boundaries", before="parser")
transcript = nlp(transcript)

matcher = Matcher(nlp.vocab)



# Preprocessing function

def is_token_allowed(token):
    return bool(token and str(token).strip() and not token.is_stop and not token.is_punct)

def preprocess_token(token):
    return token.lemma_.strip().lower()

complete_filtered_tokens = [
    preprocess_token(token) for token in transcript if is_token_allowed(token)
]


# words = [token.text for token in transcript if not token.is_stop and not token.is_punct]

#%%
displacy.render(transcript, style="dep", jupyter=True)
#%%
# print(Counter(words).most_common(5))

# print([token for token in transcript if not token.is_stop])

# Determine root word (lemma)
# for token in transcript:
#     if str(token) != str(token.lemma_):
#         print(f"{str(token):>20} : {str(token.lemma_)}")

# sentences = list(transcript.sents)
# for sentence in sentences:
#     print(sentence)

## POS tagging
# for token in transcript:
#     print(
#         f"""
# TOKEN: {str(token)}
# ====
# TAG: {str(token.tag_):10} POS: {token.pos_}
# EXPLANATION: {spacy.explain(token.tag_)}"""
#     )

