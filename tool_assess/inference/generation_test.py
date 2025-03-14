import argparse
import time
from fastchat.model import load_model

from utils.query import add_model_args, query_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--message", type=str, default="Hello! Who are you?")
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    # Pre-load model
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    start_time = time.time()
    # load message and query
    query_model(model, tokenizer, args)

    end_time = time.time()
    response_time = end_time - start_time
    minutes, seconds = divmod(int(response_time), 60)
    print(f"The model took {minutes} minutes {seconds} seconds.")