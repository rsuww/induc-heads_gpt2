# Induction Heads in GPT-2 Small

An educational walkthrough of **induction heads** — attention heads that implement in-context pattern matching of the form `[A][B] ... [A] → [B]` — reproduced in GPT-2 small using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

This notebook follows the approach laid out by **Neel Nanda** in his TransformerLens tutorials, and the original phenomenon described in **Olsson et al., "In-context Learning and Induction Heads"** (Anthropic, 2022).

## What's inside

`induction_heads.ipynb` walks through:

1. Loading GPT-2 small with TransformerLens.
2. Building **repeated random token** sequences (`random + random`) so that any in-context pattern matching cannot be memorization.
3. Confirming the model's per-token loss **drops sharply on the repeated half** — a signature of in-context learning.
4. Computing an **induction score** for every (layer, head) and visualizing the resulting heatmap over all 144 heads.
5. Identifying the strongest induction heads in GPT-2 small (typically `L5H1` and `L5H5`).
6. Visualizing the **diagonal-stripe attention pattern** with `circuitsvis`.
7. **Zero-ablating** the top induction heads and showing the loss spike on the repeated half — the causal evidence.
8. A short note on the **2-head circuit** (previous-token head + induction head, composed via K-composition).

## What is an induction head?

An attention head where the OV circuit copies and the QK circuit attends to "the token that came after the last occurrence of the current token". For sequence `... A B ... A ?`, the head:

- at the second `A`, attends back to the position of `B` (the token after the first `A`);
- copies `B` into the residual stream, increasing the logit for `B` as the next token.

This is one of the simplest known **mechanistic explanations of in-context learning** in transformers.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook induction_heads.ipynb
```

The notebook downloads GPT-2 small (~500 MB) on first run via HuggingFace. A GPU is not required — CPU works, just slower.

## Credits

- **Neel Nanda** — TransformerLens, the induction heads tutorial, and the broader [mech interp explainer](https://www.neelnanda.io/mechanistic-interpretability/).
- **Anthropic interpretability team** — the original induction heads paper (Olsson et al., 2022).
- This repo is an **educational reproduction** intended for learning, not original research.

## License

MIT.
