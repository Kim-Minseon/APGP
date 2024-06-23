import os
os.environ['TRANSFORMERS_CACHE'] = '/.cache/'
import torch
import torch.nn as nn

import pdb 
from argument import argument
from utils import read_into_list, load_model, log_folder
from pipeline import prompt_copyright_attack, seed_prompt, qa_gen
from PIL import Image
DATASET_PATH = "./Dataset/"
def main():
    args = argument()
    args.target_img_path = DATASET_PATH +"img_path_"+args.test_type+".txt"
    args.input_seq = DATASET_PATH+"keywords_"+args.test_type+".txt"
    
    # Prepare an input for the model
    img_paths, prompt_lists, described_lists, key_words = [], [], [], []

    img_paths = read_into_list(args.target_img_path, img_paths)
    key_words = read_into_list(args.input_seq, key_words)
    folder_name = log_folder(args)
    result_filename = './results/'+folder_name+'score_keyword.txt'

    if 'gpt' in args.model or 'gpt' in args.seed_prompt_model:
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    optim_llm, optim_token = load_model(args.seed_inst_optim_llm, args.deepspeed, args.master_port)   
    
    for i in range(len(img_paths)):
        target_img = Image.open(img_paths[i])
        key_word = key_words[i]
        
        args.outdir = './results/'+folder_name+str(key_word.split(',')[0])+'/'
        
        try:
            os.makedirs(args.outdir, exist_ok=True)
        except:
            if args.local_rank == 0:    
                print('save image in '+args.outdir)

        q_list, a_list = qa_gen(args, target_img, key_word, img_paths[i])
    
        if args.seed_prompt_model == 'gpt4-vision':
            prompt = seed_prompt(args, optim_llm, optim_token, target_img, key_word, img_paths[i])
        else:
            prompt = seed_prompt(args, optim_llm, optim_token, target_img, key_word)
        
        if args.local_rank == 0:
            print(f"Seed prompt of {key_word} : {prompt[0]}")
        max_prompt, score = prompt_copyright_attack(args, optim_llm, optim_token, target_img, key_word, prompt, q_list, a_list)

        if args.local_rank == 0:
            with open(result_filename, 'a') as file:
                if score == 0:
                    file.write(f"{key_word}: Blocked\n")
                else:
                    file.write(f"{key_word}: score {score} : {max_prompt}\n")
    

if __name__ == "__main__":
    main()
