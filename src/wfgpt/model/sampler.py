import torch

def sample_from_model(model, init_from, tokenizer=None):
    prompts = [
                "In West Flanders, people often say",
                "De zee es schoon vandaag,",
                "Ik goa noar de markt,"
            ]

    print("\nSampling from the", init_from, "model:")
        
    for prompt in prompts:
        if init_from == "finetuning":
                generated_text = model.generate(prompt, max_length=50, temperature=0.7, top_k=50)
                print(f"Prompt: {prompt}")
                print(f"Generated (from finetuning): {generated_text}\n")

        elif init_from == "scratch":   
                model.eval()
                with torch.no_grad():
                    idx = tokenizer(prompt, return_tensors="pt")['input_ids']
                    generated_idx = model.generate(idx, max_new_tokens=50, temperature=0.7, top_k=50)
                    generated_text = tokenizer.decode(generated_idx[0], skip_special_tokens=True)
                    print(f"Prompt: {prompt}")
                    print(f"Generated (from scratch): {generated_text}\n")
    print("Sampling completed!")

