# Roberta Zinc 480m

This is a Roberta style masked language model trained on ~480m SMILES strings from the [ZINC database](https://zinc.docking.org/) available through [Huggingface](https://huggingface.co/entropy/roberta_zinc_480m).
The model has ~102m parameters and was trained for 150000 iterations with a batch size of 4096 to a validation loss of ~0.122.
This model is useful for generating embeddings from SMILES strings.

To use, install the [transformers](https://github.com/huggingface/transformers) library:

```
pip install transformers
```

Then use the following:

```python
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorWithPadding

tokenizer = RobertaTokenizerFast.from_pretrained("entropy/roberta_zinc_480m", max_len=128)
model = RobertaForMaskedLM.from_pretrained('entropy/roberta_zinc_480m')
collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')

smiles = ['Brc1cc2c(NCc3ccccc3)ncnc2s1',
 'Brc1cc2c(NCc3ccccn3)ncnc2s1',
 'Brc1cc2c(NCc3cccs3)ncnc2s1',
 'Brc1cc2c(NCc3ccncc3)ncnc2s1',
 'Brc1cc2c(Nc3ccccc3)ncnc2s1']

inputs = collator(tokenizer(smiles))
outputs = model(**inputs, output_hidden_states=True)
full_embeddings = outputs[1][-1]
mask = inputs['attention_mask']
embeddings = ((full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1))
```
