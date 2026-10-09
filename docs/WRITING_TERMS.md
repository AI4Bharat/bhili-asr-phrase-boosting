# Writing a term list

Phrase boosting gives the decoder a small bonus whenever its hypothesis matches a term on your
list. It helps with words the model **nearly** hears but spells wrong, mostly technical and
domain words. It cannot create a word the model does not hear at all, and a bad list makes
accuracy worse. This page explains how to write a list that helps.

## What to put in

Terms the bot is expected to understand that are rare in everyday speech:

| Category | Examples |
|---|---|
| Government schemes | पीएम किसान, सन्मान निधी |
| Crops and varieties | भात, भेंडी, कपाशी |
| Pests and diseases | एच एन पी व्ही, (local names of pests) |
| Inputs | ग्लिरीसिडिया, गिरीपुष्प, ठिबक, fertiliser and pesticide names |
| Places | districts, talukas, villages, mandis |

The bot's own knowledge base or topic list is the best source: farmers ask about what the bot
talks about.

## How to write each term

1. **Devanagari, spelled the way farmers say it in Bhili**, not the textbook Marathi or Hindi
   spelling. The model writes Bhili the way its training transcripts did.
2. **Acronyms letter by letter, as spoken.** `एच एन पी व्ही`, not `HNPV` or `एचएनपीव्ही`.
3. **Numbers as words.** `दोन टक्के`, not `2 टक्के`.
4. **Variants on separate lines.** If farmers say both `कपाशी` and `कपाशीवर`, list both.
5. **One term per line.** Multi-word phrases are fine (`पीएम किसान`).

## What to leave out

- **Everyday words** (`हो`, `माहिती`, `पाणी`, `काय`, `सांगा`). They are already recognised, and
  boosting them inserts them into unrelated turns. In testing, a 211-word list mined
  automatically from the bot's speech raised WER from 33.2 to 35.0 at strength 1.0.
- **Very short words** (one or two letters). Same problem.
- **Terms nobody asks about.** Every term adds a little risk of false insertions.

50 to 200 well-chosen terms is a good size.

## Checking the list

1. `python scripts/check_terms.py my_terms.txt` catches Latin letters, digits, generic and
   duplicate terms. Fix every ERROR; read every WARNING.
2. `python scripts/evaluate.py --terms my_terms.txt --manifest test.jsonl --report report.md`
   measures the list on audio with reference transcripts and gives each term a verdict:
   - **KEEP**: boosting makes it come out right more often.
   - **DROP**: boosting writes it where it was not said more often than it fixes it. Remove
     or respell it.
   - **CHECK**: no effect on this data, usually because the term is absent from the references.
3. `python scripts/transcribe.py --terms my_terms.txt --compare <real calls>` shows every
   transcript that boosting changes, so a Bhili speaker can judge real calls.

## Strength

Keep `--alpha 0.5` (the default). In testing, raising it to 1.0 made `ठिबक` and `हिरव्या`
appear in turns where nobody said them, and 2.0 raised WER by more than 3 points. Raise it only
if `evaluate.py` shows the gain is worth the extra insertions.
