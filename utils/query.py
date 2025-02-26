import torch

from fastchat.model import load_model, get_conversation_template, add_model_args

from utils.util import load_messages, save_results

torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@torch.inference_mode()
def query_model(model, tokenizer, args):
    # Build the prompt with a conversation template
    messages = load_messages()
    results = []
    for msg in messages:
        conv = get_conversation_template(args.model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Run inference
        inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
        output_ids = model.generate(
            **inputs,
            do_sample=True if args.temperature > 1e-5 else False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        # Get outputs
        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        # Print results
        print(f"{conv.roles[0]}: {msg}")
        print(f"{conv.roles[1]}: {outputs}")
        result = {
            # "index": conv.roles[0],
            "question": msg,
            "response": outputs
        }
        results.append(result)

    save_results(results, args.model_path)