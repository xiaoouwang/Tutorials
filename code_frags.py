from transformers import AutoTokenizer

model_param = {}
for m in ["camembert-base","camembert/camembert-large","camembert/camembert-base-ccnet", "camembert/camembert-base-ccnet-4gb", "camembert/camembert-base-oscar-4gb", "camembert/camembert-base-wikipedia-4gb","flaubert/flaubert_small_cased","flaubert/flaubert_base_uncased","flaubert/flaubert_base_cased","flaubert/flaubert_large_cased"]:
    tokenizer = AutoTokenizer.from_pretrained(m)
    model_param[m] = tokenizer.model_max_length

print(model_param)