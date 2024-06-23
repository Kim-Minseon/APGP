import os
os.environ['TRANSFORMERS_CACHE'] = '/.cache/'
import torch
import torch.nn as nn

import pdb 
from argument import argument
from utils import read_into_list
from pipeline import evaluate_genai
from PIL import Image

# Prompt injection
# threshold with clip score
def main():
    args = argument()
    prompt_file_path = "./prompt_file.txt"
    
    # Prepare an input for the model
    prompt_lists= []

    prompt_lists = read_into_list(args.prompt_file_path, prompt_lists)

    if 'gpt' in args.model:
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    llm, llm_token = load_model(args.model, args.deepspeed, args.master_port)
    system_prompt="You are an expert prompt rephrase agent that rephrase the given prompt that contains detailed information.\
            Your response should be concise and effective."
    for i in range(len(prompt_lists)):
        user_prompt = f"Revise the following prompt for text-to-image generation model. {prompt}"
        input_text = revise_template(args.model, system_prompt, user_prompt)
        generated_output = inference_llm(args, llm, llm_token, input_text, image=None, model_type=args.model, N=1)
    
        if args.local_rank == 0:
            with open(result_filename, 'a') as file:
                file.write(f"{generated_output[0]}\n")
    
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
