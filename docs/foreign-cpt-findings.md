# Foreign-distribution CPT findings (2026-05-27/28)

## Question

The Track 1 SFT default (`HuggingFaceH4/ultrachat_200k` + `Qwen3.5-4B-Base`)
produces a `baseline_eval_loss − final_eval_loss` of roughly **+0.05** over a
30-minute LoRA fine-tune. That is barely above noise. We wanted a dataset that
gives a **pronounced** loss-drop signal — large enough that experiments can
distinguish good and bad training choices clearly, and large enough that the
"score" number tells a meaningful story.

## Hypothesis

The drop is small because ultrachat is general English chat: very close to
Qwen's pretraining distribution, so its baseline loss is already low (~0.13)
and there is little headroom for training to improve it. Picking a dataset
that is **demonstrably outside** the pretraining distribution should give a
high baseline loss and a much larger achievable drop.

## Candidates considered

| Candidate | Tokens | Decipherment | Status |
|---|---|---|---|
| **ConlangCrafter** synthetic conlang (one of 64 specs in [malper/ConlangCrafter](https://huggingface.co/datasets/malper/ConlangCrafter), arXiv [2508.06094](https://arxiv.org/abs/2508.06094)) | ~11M (generated) | n/a — fully synthetic | **Selected** |
| **SumTablets** Sumerian cuneiform transliterations ([colesimmons/SumTablets](https://huggingface.co/datasets/colesimmons/SumTablets), arXiv [2602.22200](https://arxiv.org/abs/2602.22200)) | ~10–15M (train split, 82,452 rows, 30.2M chars) | Deciphered | **Selected** |
| **FineMath** baseline (`HuggingFaceTB/finemath`) | — | n/a | Anchor |
| Linear A | ~7,400 signs total across 1,427 inscriptions | **Undeciphered** | **Rejected.** 3+ orders of magnitude too small; without decipherment there is no grammar regularity to learn, so any loss drop would be memorization. |
| Other undeciphered scripts (Indus, Rongorongo, Voynich, Proto-Elamite, Phaistos Disc) | thousands of signs | Undeciphered | **Rejected** for the same reasons. |
| Tibetan (TIBSTC, `pkupie/mc2_corpus`) | 11B | Living | **Rejected** — well-represented in modern pretraining, not OOD enough. |

## Method

CPT (continued pretraining), not SFT. Raw text packed into `seq_len`=4096
blocks, all-token loss. Identical hyperparameters across all three candidates
(LoRA rank 32, AdamW fused, lr 2e-4, micro_batch_size 8, flex-attention,
`max-autotune-no-cudagraphs` compile, 64-block held-out eval).

Each candidate gets a Track 2 (5-minute) run. The winner gets a Track 1
(30-minute) run.

The synthetic conlang corpus was generated one-shot via Vertex AI Gemini 3.5
Flash (`scripts/synthesize_conlang_cpt.py`):

- One ConlangCrafter spec (`bd412d52`, DeepSeek-R1-generated, 131 lexicon
  words, polysynthetic, IPA with tones and clicks) as the system prompt.
- Rotated topic seeds across 50 prompts for diversity.
- Per-chunk quality gate: lexicon-overlap minimum (substring match, ≥50%),
  English-word ratio maximum (≤5%), minimum length (400 chars). Failed chunks
  retried up to 2× with a different topic.
- `thinking_config=ThinkingConfig(thinking_budget=0)` — without this Gemini
  3.5 Flash silently burns the entire output budget on thoughts and returns
  empty text. **Important debugging finding.**
- `max_output_tokens=8192` with headroom — when the model hits MAX_TOKENS,
  the truncated final part can have `text=None` and `resp.text` returns `""`
  even after thousands of generated tokens. Always leave slack.
- Async concurrency 32 against Vertex on `gemini-3.5-flash` (global endpoint).
- Result: 3,077 chunks, 10.99M output tokens, 13.26M chars, 27 minutes,
  0 final rejects (all gated chunks succeeded on retry).
- Cost: ~$5–20 (Flash pricing).
- Published: [`TearedModels/conlangcrafter-cpt-bd412d52`](https://huggingface.co/datasets/TearedModels/conlangcrafter-cpt-bd412d52).

## Results

### Track 2 — 5 minutes (3-way sweep)

| Dataset | Eval-loss drop | Baseline | Final | Steps | Tokens |
|---|---:|---:|---:|---:|---:|
| FineMath (legacy CPT) | **−0.034** ❌ | 1.431 | 1.466 | 101 | 3.31M |
| ConlangCrafter (synthetic) | **+0.510** ✅ | 0.854 | 0.345 | 101 | 3.31M |
| SumTablets (Sumerian) | **+1.092** ✅✅ | 1.946 | 0.855 | 99 | 3.24M |

Both foreign datasets dwarf the FineMath signal (and the prior Hermes-SFT
Track 1 record of +0.052). Sumerian gives the largest absolute drop;
ConlangCrafter gives the largest **relative** drop (60% of baseline loss
eliminated) and the lowest final loss.

### Track 1 — 30 minutes (ConlangCrafter, seed 1337)

| Metric | Value |
|---|---:|
| eval_loss_drop | **+0.540** |
| baseline_eval_loss | 0.854 |
| final_eval_loss | 0.315 |
| steps | 604 |
| tokens | 19.79M |
| supervised tok/s | 10,989 |
| peak GPU util | 100% |

[`records/track_1_30min/2026-05-28_ConlangCrafter_CPT_Track1_seed1337/`](../records/track_1_30min/2026-05-28_ConlangCrafter_CPT_Track1_seed1337/)
contains the full snapshot ([Modal run](https://modal.com/apps/tear-labs-43657/main/ap-lv4L5notEjWXBIJhrpNcOe)).

Roughly **10× the previous best Track 1 signal** (Hermes-SFT GraLoRA at
+0.052).

## Interpretation

1. **Distribution distance is the dominant lever for loss-drop magnitude.**
   Going from in-distribution (FineMath, baseline 1.43) to clearly OOD
   (Sumerian, baseline 1.95) or fully novel (synthetic conlang, baseline
   0.85) changes the achievable drop by 1–2 orders of magnitude. Optimizer,
   adapter, and schedule choices — which dominated prior iteration logs —
   are much smaller effects than the dataset choice.

2. **The synthetic conlang has a counter-intuitively *low* baseline loss
   (~0.85) despite being novel.** Best hypothesis: the IPA/tone-mark
   characters tokenize into many small sub-character pieces that the
   tokenizer's distribution model can predict relatively well from local
   bigram statistics, plus the corpus has repetitive lexical structure (root
   words like `k'u`, `hun`, `wa.la` recur frequently). The model can predict
   "next sub-character within a known root" without knowing the language. So
   the *absolute* drop is smaller than Sumerian's, even though the conlang
   is more OOD.

3. **Sumerian has a high baseline (1.95) because Latin-alphabet
   transliteration with subscripts like `uri₅{ki}ma` tokenizes into long,
   unfamiliar Qwen sequences.** The drop is large because the model genuinely
   learns the transliteration structure (case endings, determinatives like
   `{ki}`/`{d}`, surface/column markers) from very few epochs.

4. **5 minutes captures ~95% of the drop on the conlang.** Track 2
   (5min) gave +0.510; Track 1 (30min) gave +0.540 — diminishing returns
   after the model fits the lexical inventory. This suggests the conlang
   corpus may not be the best fit for a 30-min budget; a *harder* OOD source
   would let more of the 30-minute budget translate into eval-loss progress.

5. **Both candidates pass the "+0 drop" record-eligibility rule by huge
   margins** with no tuning. Future record iterations on these datasets
   should focus on the systems side (better optimizer, higher tok/s, better
   adapter init), not on chasing additional dataset shift.

## Caveats

- **Single-seed Track 1** for the conlang. Promoting to a record claim under
  the README's p<0.01 rule needs seeds 2027 and 4099. Commands are in the
  README's Track 1 table.
- **Tokenization sensitivity.** The conlang's low baseline loss is partly a
  tokenization artifact (sub-character IPA pieces). A different base model
  with a different tokenizer would likely show different baseline numbers
  but the same general "drop is much larger than ultrachat" story.
- **Train/eval split for the conlang.** The held-out eval blocks are
  constructed from the unshuffled stream's leading documents (same as the
  FineMath path), which means our eval is drawn from the same chunk
  distribution as training. For a truly clean evaluation, generate a separate
  held-out set with `--target-tokens` smaller and a different seed.
- **The conlang corpus is one specific language (`bd412d52`).** Different
  ConlangCrafter languages would give different baseline/drop numbers. We
  picked the longest-spec DeepSeek-R1 language; nothing about the choice was
  optimized for "easiest learning."
- **SumTablets baseline run was not retried for noise.** The +1.09 drop is a
  single observation. Repeat with multiple seeds before treating it as a
  stable headline.
- **Catastrophic forgetting on the base task.** We did not evaluate whether
  the CPT-on-conlang or CPT-on-Sumerian adapters degrade English performance.
  That matters if these adapters are intended as anything other than
  benchmark-loss-drop targets.

## How to reproduce

```bash
# 1. Regenerate the conlang corpus and push to your HF account.
source ~/.config/.env.global   # provides VERTEXAI_PROJECT etc.
uv run python scripts/synthesize_conlang_cpt.py --target-tokens 10_000_000 --concurrency 32
uv run python scripts/push_conlang_dataset.py data/conlang_cpt/<language_id>

# 2. Track 2 (5-min) sweep.
export CONLANG_DATASET_ID=<your-hf-user>/conlangcrafter-cpt-<language_id>
./run.sh conlang-track2
./run.sh sumerian-track2     # uses colesimmons/SumTablets + --cpt-text-field transliteration
./run.sh track2              # FineMath baseline

# 3. Track 1 record runs.
./run.sh conlang-track1 --seed 1337 --record-description "ConlangCrafter CPT seed1337" --record-contributors "@you"
./run.sh conlang-track1 --seed 2027 --record-description "ConlangCrafter CPT seed2027" --record-contributors "@you"
./run.sh conlang-track1 --seed 4099 --record-description "ConlangCrafter CPT seed4099" --record-contributors "@you"
```

## References

- ConlangCrafter (Alper et al., 2026), arXiv [2508.06094](https://arxiv.org/abs/2508.06094).
- SumTablets (Simmons, 2024), arXiv [2602.22200](https://arxiv.org/abs/2602.22200).
- Linear A Digital Corpus (Salgarella & Castellan, 2015), [aclanthology W15-3715](https://aclanthology.org/W15-3715.pdf).
- "All Code, No Thought": ciphered reasoning is OOD (Oct 2025), arXiv [2510.09714](https://arxiv.org/pdf/2510.09714).
