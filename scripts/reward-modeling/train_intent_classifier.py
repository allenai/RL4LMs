from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from rl4lms.data_pools.custom_text_generation_pools import DailyDialog
from datasets.arrow_dataset import Dataset

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


def main():

    # label
    label = "intent"
    
    # results folder
    results_folder = f"./results/{label}"

    # data pool
    train_dp = DailyDialog.prepare("train", 1)
    val_dp = DailyDialog.prepare("val", 1)

    # train and val dataset
    ds_train = get_dataset(train_dp, label)
    ds_test = get_dataset(val_dp, label)

    model_name = "cardiffnlp/twitter-roberta-base-emotion"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(examples):
        outputs = tokenizer(examples['text'], truncation=True)
        return outputs

    tokenized_ds_train = ds_train.map(tokenize, batched=True)
    tokenized_ds_test = ds_test.map(tokenize, batched=True)

    def compute_metrics(eval_preds):
        metric = load_metric("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(num_train_epochs=10,
                                      output_dir=results_folder,
                                      # push_to_hub=True,
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=64,
                                      evaluation_strategy="steps",
                                      save_strategy='steps',
                                      logging_steps=20,
                                      save_total_limit=1,
                                      save_steps=100,
                                      lr_scheduler_type="constant",
                                      learning_rate=1e-6,
                                      )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer,
                      data_collator=data_collator,
                      args=training_args,
                      train_dataset=tokenized_ds_train,
                      eval_dataset=tokenized_ds_test,
                      compute_metrics=compute_metrics)

    trainer.train(resume_from_checkpoint=True)

if __name__ == '__main__':
    main()

