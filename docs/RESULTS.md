# Measured results (2026-10)

All numbers are for `ai4bharat/bhili-asr-nemotron-600m` with the Bhili head selected, on the
Bhili test set (4,130 utterances; 1,375 of them are sentences that do not appear in the
training data). WER in percent, lower is better.

## Decoding and boosting

| Setup | WER all | WER, sentences not in training | Conversational subset | Term false insertions |
|---|---|---|---|---|
| greedy, no boosting | 33.16 | 57.16 | 71.85 | 165 |
| greedy + 16 terms, strength 0.5 | 33.28 | 57.45 | 72.19 | 174 |
| **beam 4, no boosting** | **32.95** | **56.91** | 71.42 | 164 |
| **beam 4 + 16 terms, strength 0.5 (default)** | 33.06 | 57.19 | 71.75 | 184 |
| beam 4 + 16 terms, strength 1.0 | 33.17 | 57.20 | 71.81 | 264 |
| beam 4 + 16 terms, strength 2.0 | 36.43 | 62.44 | 78.47 | 2,550 |
| greedy + 211 auto-mined words, strength 1.0 | 34.98 | 60.01 | 74.02 | 1,056 |
| greedy + token 4-gram LM, weight 0.2 | 32.96 | 57.17 | 71.97 | 166 |

"Term false insertions" counts boosted words written where the reference does not have them.
The 16 terms are `terms/example_terms.txt`.

## On real field calls

Farmer turns from field-test calls, decoded with beam 4 + the 16 terms at strength 0.5:

| Farmer said | Without boosting | With boosting |
|---|---|---|
| HNPV बद्दल माहिती द्या | एचएम बद्दल माहिती द्या | एच एन पी व्ही बदल माहिती द्या |
| HNPV व्हायरस … मोड ऑफ ॲक्शन | एच एमपी व्ही व्हायरस … | एच एन पी व्ही व्हायरस … |

Gliricidia (ग्लिरीसिडिया) and गिरीपुष्प also came out correctly more often. **PM-Kisan was not
recovered at any usable strength**: the model never produces anything close to "पीएम किसान" on
those calls, so there is nothing to boost. Casual conversational speech is also out of reach of
boosting. Both need training data.

The 16 terms were chosen from the field complaints, so the field results are a best case.
Measure any new list with `scripts/evaluate.py` before deploying it.

## Conclusions

- Use beam search (width 4). It is slightly more accurate than greedy decoding on its own.
- A short curated list at strength 0.5 fixes technical terms the model nearly hears, at a cost
  of about +0.1 WER overall.
- Lists mined automatically, or high strengths, make accuracy worse.
- An n-gram language model on the training text did not help on unseen sentences.

## The evaluation tool on these terms

`scripts/evaluate.py --terms terms/example_terms.txt` on the first 400 test utterances
(beam 4, strength 0.5): WER 33.96 → 33.90. It marks `ठिबक` as **DROP** (written twice where
the references do not have it, never needed). This matches what we saw on field calls, where
`ठिबक` was the term most often inserted where nobody said it. Most other terms are **CHECK**
because they rarely occur in the read-speech test set; field calls with reference transcripts
are the better place to judge them.
