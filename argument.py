import argparse
import json
def argument():
    parser = argparse.ArgumentParser("APGP", add_help=False)
    
    # args for logging
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--file_suffix", type=str, default='',
                        help="file suffix")
    
    # args for APGP
    parser.add_argument("--test_type", type=str, default='ip', choices=['art', 'logo', 'character', 'product', 'architecture', 'all', 'example'], help='dataset type')    
    
    # args for model type
    parser.add_argument("--model", type=str, default='gpt3.5',
                        help="LM model type", choices=['gpt3.5', 'gpt4-vision'])
    parser.add_argument("--generation_blackbox", type=str, default='dalle3', choices=['dalle2', 'dalle3'])
    parser.add_argument("--seed_prompt_model", type=str, default='llava', choices=['llava', 'gpt4-vision'])
    parser.add_argument("--seed_inst_optim_llm", type=str, default='gpt3.5',
                        help="Seed prompt instruction otimzier LM model type", choices=['gpt3.5'])
    
    #args for seed description stage
    parser.add_argument('--seed_list', type=lambda a: json.loads('['+a.replace(" ",",")+']'), default="0 1 2", help="List of values")                                              
    parser.add_argument("--seed_automated",  action='store_true',
                        help="seed prompting automated or not version") 
    parser.add_argument("--seed_automated_strategy", type=str, default='ours',
                        help="Seed prompt optimzing type")
    parser.add_argument("--unchange_update_num", type=int, default=5,
                        help="Automated prompt tuning iteration number")   
    
    # args for rephrase stage
    parser.add_argument("--rephrase_automated_strategy", type=str, default='ours',
                        help="Revise prompt optimzing type")
    parser.add_argument("--rephrase_cand_num", type=int, default=3,
                        help="Automated prompt revision number")  
    parser.add_argument("--rephrase_iter", type=int, default=3,
                        help="Rephrase iteration number")
    parser.add_argument("--num_rephrase", type=int, default=1,
                        help="Rephrased sentences number")

    # args for ablation study
    parser.add_argument("--wo_qa", action="store_true")
    parser.add_argument("--qa_ablation", type=int, default=1, help="qa type")
    
    # args for deepspeed
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument('--master_port', type=str, default="28900")
    parser.add_argument('--local_rank', type=int, default=0)
    
    parser.add_argument("--n_samples", type=int, default=1,
                            help="how many samples to produce for each given prompt. A.k.a. batch size")
    
    return parser.parse_args()

