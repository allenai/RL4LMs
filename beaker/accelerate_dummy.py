from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader


def main():
    raw_datasets = load_dataset("glue", "mrpc")
    #checkpoint = "bert-base-uncased"
    #checkpoint_tok = '/net/nfs.cirrascale/mosaic/raja/llama/llama-tokenizer'
    #checkpoint = '/net/nfs.cirrascale/mosaic/raja/llama/llama-7b'
    #tokenizer = AutoTokenizer.from_pretrained(checkpoint_tok)
    checkpoint = 'bigscience/bloom-7b1'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)


    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")



    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    accelerator = Accelerator()

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model = accelerator.prepare(model)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    train_dl, eval_dl, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, optimizer
    )
    print("ACCELERATE", accelerator.is_main_process, accelerator.process_index)

    num_epochs = 50
    num_training_steps = num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

if __name__ == '__main__':
    main()