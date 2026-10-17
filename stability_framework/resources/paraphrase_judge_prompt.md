# Paraphrase substitution judge prompt (version 1)

The judge is Claude Haiku (`claude-haiku-5-5`), run through Claude Code subagents. Batches are 100 items; verdicts are stored in `paraphrase_judgments.jsonl`.

Each input line has:
- an id;
- a text in which one word is marked [[like this]];
- the word;
- a proposed substitute.

Instructions given to the judge:

> For each item, decide whether replacing the marked word with the substitute, in this exact context, preserves the text's meaning. Answer:
> - Y if the substitute keeps the same sense and truth conditions, does not noticeably change sentiment, intensity or topic, and the result is still grammatical and natural enough to read as a paraphrase;
> - N otherwise. With N, give one code: S (wrong sense or meaning change), G (ungrammatical or wrong part of speech), T (changes tone or intensity), O (other).
>
> Texts may be lowercased and tokenised (movie reviews) or news headlines followed by their lead sentence. Judge each item independently and strictly: if in doubt, answer N.

Output: one JSON object per item, `{"id": ..., "v": "Y"}` or `{"id": ..., "v": "N", "c": "S"}`.

## Clause-swap items (same prompt version; used from the full run on)

Some input lines have `reordered` instead of `word`/`substitute`: two clauses of the text swapped. The judge answers Y if the reordered text says the same thing (same claims, same relations between them, still grammatical), otherwise N with a code.
