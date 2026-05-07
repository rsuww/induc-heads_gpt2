# Induction Heads in GPT-2 Small

a walkthrough of **induction heads** :
what are they? theyre attention heads (in transformer architecture) that implement in-context pattern matching of the form '[a][b] -> [a]->[b]' using transformerlens[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

following tutorial from neel nanda

## What's inside
1. load GPT-2 small (legendary) with TransformerLens
2. build **repeated random token** sequences (`random + random`) so that any in-context pattern matching cannot be memorization
3. confirm the model's per-token loss **drops sharply on the repeated half** 
4. compute an **induction score** for every (layer, head) and visualizing the resulting heatmap over all 144 heads (yay for heatmaps)
5. identify strongest induction heads in GPT-2 small
6. visualize the **diagonal-stripe attention pattern** with `circuitsvis`
7. ablaiton of the top induction heads and showing the loss spike on the repeated half


this is the basic start known to **mechanistic explanations of in-context learning** in transformers

## Run it

```bash
pip install -r requirements.txt
jupyter notebook induction_heads.ipynb
```

i downloaded GPT-2 small (~500 MB) on first run, use a mac tbh

## Credits
neel nanda and transformerlens
