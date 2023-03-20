from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

def main():
    checkpoint = "/net/nfs.cirrascale/mosaic/raja/llama/llama-13b"
    config = AutoConfig.from_pretrained(checkpoint)
    from accelerate import load_checkpoint_and_dispatch

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model = load_checkpoint_and_dispatch(
        model, "/net/nfs.cirrascale/mosaic/raja/llama/llama-7b", device_map="auto", no_split_module_classes=["LlamaDecoderLayer"]
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/net/nfs.cirrascale/mosaic/raja/llama/llama-tokenizer")
    inputs = tokenizer("Hello, my name is", return_tensors="pt")
    inputs = inputs.to(0)
    output = model.generate(inputs["input_ids"])
    tokenizer.decode(output[0].tolist())

if __name__ == "__main__":
    main()