# Changelog

## 1.1.0: Alignment with the published protocol

This release fixes defects in the code first published alongside the paper
(commit `cc9153f`) and brings the implementation in line with the protocol the
paper describes. **Several fixes change the numbers the code produces**, so
results from 1.1.0 are not expected to match Tables I and II exactly. Each item
below lists which results it affects.

### Explanation methods

- **Integrated Gradients ran with dropout enabled.** `model.train()` was called
  before attribution, so IG was stochastic (top-5 tokens changed between
  identical calls). The paper states IG is deterministic. IG now runs in eval
  mode. *Affects: all IG rows.*
- **Integrated Gradients explained a different function for DistilBERT.** The
  attribution forward fed the `[CLS]` state straight into `classifier`,
  skipping `pre_classifier` and its ReLU (explained logits `[0.06, -0.07]` vs.
  model logits `[-4.34, 4.69]` for the same input). IG now attributes through
  the model's own forward pass for every architecture. *Affects: DistilBERT IG
  rows (Tables I and II).*
- **The IG baseline was the `[PAD]` embedding, not a zero vector** as the paper
  states. It is now a zero vector in embedding space. The per-explanation IG
  convergence delta is now reported so the 50-step integration error is
  visible. *Affects: all IG rows.*
- **SHAP failed on every input** with the paper-era stack (transformers 4.56,
  shap 0.48): SHAP passes a numpy array of strings, which tokenizers reject,
  and the per-sample error handling silently dropped every sample. *Affects:
  all SHAP rows could not be produced by the published code in this
  environment.*

### Models

- **T5 used a randomly initialised classification head.** The paper describes
  T5 classifying by conditional generation with class labels as targets. The
  code instead attached an untrained `Linear(hidden, 2)` head (37-48% accuracy
  on SST-2, depending on the torch seed). T5 now scores the label words
  "negative"/"positive" at the decoder's first step, using the prompt the
  checkpoint was fine-tuned with; no parameters are added. Accuracy on the
  SST-2 validation set is 93.6%, and predictions match free generation from
  the checkpoint. *Affects: all T5 rows (Table I).*
- **Experiment 4 used a 2-class model on 4-class AG News.**
  `jerrybuilds/distilbert-ag_news` has two output labels (46% accuracy on
  AG News). It is replaced by `textattack/distilbert-base-uncased-ag-news`
  (4 labels, 94.8% on the full AG News test set), from the same training
  family as Experiment 3's RoBERTa. *Affects: all DistilBERT AG News rows
  (Table II).*

### Assessment protocol

- **Correct predictions only** (paper, Section III.D) was not implemented;
  misclassified instances were evaluated too (78 of 872 SST-2 instances for
  DistilBERT). They are now removed before stress testing, and the filter
  summary is saved with the results. *Affects: all rows.*
- **Per-class results for multi-class tasks** (Section III.D) were not
  computed. They are now part of the results and the AG News stability cards.
- **Runner stress-test settings were overwritten by built-in defaults**, and
  seed-variation LIME re-ran with 1000 samples regardless of the runner's
  setting. Both now honour the configuration.
- **LIME explanations depended on call order** (its random state advanced
  across texts). LIME is now seeded per call, so an explanation depends only
  on the text and seed. *Affects: LIME rows.*
- **The Experiment 2 and 4 runners did not use the paper's settings** (200 and
  500 instance caps; IG 25 steps and LIME 500 samples for Experiment 2). All
  four runners now share the paper's configuration. *Affects: T5 and DistilBERT
  AG News rows.*
- The Experiment 3 runner crashed with a `KeyError` after saving its results.

### Word-level alignment of attributions

Metrics compare explanations word by word, but the word keys were built
inconsistently. *Affects: all rows, particularly preprocessing and T5.*

- Repeated words overwrote each other.
- The SentencePiece word-start marker was treated as a continuation, gluing
  T5 words together (`"thefilmis"`).
- Keys were case-sensitive, so an uppercased variant shared no words with the
  original even when the explanation was identical.
- SHAP keys kept trailing whitespace and special tokens; LIME was capped below
  its number of features.

All explainers now map token scores onto the whitespace-delimited words of the
input by character offset, and key each occurrence by its lowercased form
(`word`, `word#2`, ...).

### Semantic paraphrasing

- **Paraphrases were not reproducible.** Synonyms were drawn from a Python
  `set`, whose order changes between processes. *Affects: all paraphrasing rows.*
- **Synonym substitution replaced function words** with unrelated WordNet
  senses ("a" → "adenine", "angstrom"). Only base-form content words are now
  substituted, with same-part-of-speech synonyms.
- **One category filled every slot.** Up to four paraphrases are kept, and
  synonym variants always came first, so reordering and simple transformations
  were almost never used. Variants are now taken round-robin across the three
  categories.
- **Reordering shuffled words**, breaking grammaticality. It now swaps clauses
  joined by a comma or "and", as the paper describes.
- The paraphrase cache moved to `~/.cache/stability_framework` and its key now
  includes a generator version, so caches from older code are never reused.

### Reporting and reproducibility

- Stability cards report one row per explainer and stress test, with J@5, ρ
  and AD (%) as mean ± sd as in the paper's tables. The previous card averaged
  across explainers, reported the un-normalised attribution difference, and
  showed flip-rate as "N/A".
- Dependencies are resolved as of 2025-09-30 (`exclude-newer`) and locked in
  `uv.lock`, which was previously gitignored.
- Runner outputs go to `results/` in the repository, regardless of the working
  directory.
- Stratified dataset sampling and `max_samples` subsampling use generators
  seeded with the experiment seed.
