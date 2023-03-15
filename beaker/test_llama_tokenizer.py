from transformers import LLaMATokenizer
import numpy
import torch

tokenizer = LLaMATokenizer.from_pretrained('../llama/')
print(tokenizer.eos_token_id, tokenizer.bos_token_id)
print(tokenizer.sp_model.get_piece_size())
tokenizer.pad_token_id = tokenizer.eos_token_id
print('ID:', tokenizer._convert_id_to_token(1353))
#TYPE ERROR 1353
# TYPE ERROR 2159
tok = tokenizer(
    ["<s> The primary use of LLaMA is research on large language models, including </s>",
"<s> The </s>"],
    padding=True,
    return_tensors="pt",
    add_special_tokens=False,
    device='cuda'
)

#/opt/miniconda3/lib/python3.9/site-packages/transformers/generation/utils.py:1374: UserWarning: You are calling .generate() with the `input_ids` being on a device type different than your model's device. `input_ids` is on cuda, whereas the model is on cpu. You may experience unexpected behaviors or slower generation. Please make sure that you have put `input_ids` to the correct device by calling for example input_ids = input_ids.to('cpu') before running `.generate()`.

print(tok['input_ids'].cuda())
print(tokenizer.convert_tokens_to_string(tok['input_ids'][0].cpu().tolist()))