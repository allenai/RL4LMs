from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import numpy as np
from rl4lms.data_pools.custom_text_generation_pools import DailyDialog
from datasets.arrow_dataset import Dataset
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

def get_batch(samples, batch_size: int):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix : current_ix + batch_size]
        yield current_batch
        current_ix += batch_size

def get_dataset(datapool, label="intent"):
    # get the training data in text, label format
    texts = []
    labels = []
    for sample, _ in datapool:

        history = sample.prompt_or_input_text.split(DailyDialog.EOU_TOKEN)
        history = [utt for utt in history if utt != ""]
        last_utterance = history[-1]

        # just consider the utterance
        input_text = last_utterance + DailyDialog.EOU_TOKEN + sample.references[0]
        
        texts.append(input_text)
        labels.append(sample.meta_data[label][0]-1)
    
    print(np.unique(labels, return_counts=True))

    dataset = Dataset.from_dict(
            {
                "text": texts,
                "labels": labels
            },
            split="train"
        )
    return dataset

tokenizer = AutoTokenizer.from_pretrained("rajkumarrrk/roberta-daily-dialog-intent-classifier")
model = AutoModelForSequenceClassification.from_pretrained("rajkumarrrk/roberta-daily-dialog-intent-classifier")

# data pool
train_dp = DailyDialog.prepare("train", 1)
val_dp = DailyDialog.prepare("val", 1)

# train and val dataset
ds_train = get_dataset(train_dp, "intent")
ds_test = get_dataset(val_dp, "intent")


all_pred_labels = []
all_target_labels = []
batches = list(get_batch(ds_test, 10))
for batch in tqdm(batches):
    encoded = tokenizer(
        batch["text"],
        return_tensors="pt",
        truncation=True,
        padding=True)
    with torch.no_grad():
        outputs = model(input_ids=encoded.input_ids,
                                attention_mask=encoded.attention_mask)
        pred_labels = torch.argmax(outputs.logits, dim=1).tolist()
        all_pred_labels.extend(pred_labels)
        all_target_labels.extend(batch["labels"])

print(classification_report(all_target_labels, all_pred_labels))