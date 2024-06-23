import os
os.environ['TRANSFORMERS_CACHE'] = '/.cache/'
import torch
import torch.nn as nn

import pdb 
from argument import argument
from utils import read_into_list, similarity_score, revise_template, load_model
from utils import gen_qa_list, qa_parse
from inference import inference_llm
from PIL import Image
DATASET_PATH = "./Dataset/"

import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import argparse

def vlm_inference(args, target_img, img_path, system_prompt, user_prompt):
    if args.vlm_model == 'gpt4-vision':
        # OpenAI API Key
        from openai import OpenAI
        import base64
        import requests
        client = OpenAI()
        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    
    if args.vlm_model == 'llava':
        llava, llava_tokenizer = load_model('llava', args.deepspeed, args.master_port)
        
    input_text = revise_template(args.vlm_model, system_prompt, user_prompt)

    if args.vlm_model == 'llava':
        output_seq_text = inference_llm(args, llava, llava_tokenizer, input_text, image=target_img, model_type='llava', N=1)
    elif args.vlm_model == 'gpt4-vision':
        # Getting the base64 string
        target_img_4_gpt = encode_image(img_path)
        if args.local_rank == 0:
            
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": input_text['system']},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_text['user']},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{target_img_4_gpt}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            output_seq_text = [response.choices[0].message.content]
        else:
            output_seq_text = ['_']
        output = [None for _ in range(torch.cuda.device_count())]
        torch.distributed.all_gather_object(output, output_seq_text[0])
        output_seq_text = [output[0]]
    else:
        raise NotImplementedError
    return output_seq_text
    
def qa_gen_eval(args, target_img, img_path= None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Q_list = []
    #for type_qa in ['question', 'ans']:
    system_prompt, user_prompt = gen_qa_list(n=args.qa)
    
    output_seq_text = vlm_inference(args, target_img, img_path, system_prompt, user_prompt)
    
    Q_list, A_list = qa_parse(output_seq_text[0])
    return Q_list, A_list

def QA_eval_llm(args, Q_list, A_list, respond_list):
    evaluator, evaluator_token = load_model(args.eval_llm, args.deepspeed, args.master_port) 
    score = 0
    system_prompt = f"You are an expert evaluator agent based on the given question and answer. \
        You take a respond as input and evaluate whether it is correct or not as output."

    for n in range(len(A_list)):
        user_prompt = f"Question is \"{Q_list[n]}\" and the correct answer is {A_list[n]}.\n"
        user_prompt += f"My respond: {respond_list[n]}\n"
        user_prompt += f"Your task is to evaluate my respond based on the question and correct answer. Write <CORRECT> if it is correct, write <WRONG> if it is incorrect. And provide the reason of your evaluation.\n"
    
        input_text = revise_template(args.eval_llm, system_prompt, user_prompt)
        
        score_txt = inference_llm(args, evaluator, evaluator_token, input_text, image=None, model_type=args.eval_llm, N=1)
        if 'CORRECT' in score_txt[0]:
            score += 1
    return score
        
def QA_score_ans(args, img, img_path, Q_list, A_list):
    
    system_prompt = f"You are an expert vision-language model agent respond based on the given image. \
        You takes a image and question as an input and respond correct answer as output."
    user_prompt = f"Your task is to respond on following questions based on the context. Respond with answers in between <ANSWER> and </ANSWER>, eg: \
    1. <ANSWER>ANSWER 1</ANSWER>\n \
    2. <ANSWER>ANSWER 2</ANSWER>\n \
    3. <ANSWER>ANSWER 3</ANSWER>\n \
    \n"
    for n in range(len(Q_list)):
        user_prompt += f'Question 1: {Q_list[n]}\n'

    respond_output = vlm_inference(args, img, img_path, system_prompt, user_prompt)
    
    if 'ANSWER' in respond_output[0]:
        respond_list = respond_output[0].split('<ANSWER>')[1:]
        for n in range(len(respond_list)):
            respond_list[n] = respond_list[n].split('</ANSWER>')[0]
    else:
        respond_list = [""]*len(A_list)
    return respond_list

def extract_feat(samples, model):
    
    features = None
    samples = samples.cuda(non_blocking=True)
    
    feats = model(samples).clone()

    return feats

def load_model_dino(args):
    import vision_transformer as vits
    import dino_utils
    # ============ building network ... ============
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    dino_utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    return model

def args_parse():
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    
    parser.add_argument("--eval_type", default='rep', type=str, help="eval_type: rep / QA")
    parser.add_argument("--eval_file", type=str, default='./gen_img_path.txt', help="qa num")
    
    parser.add_argument("--vlm_model", type=str, default='llava')
    parser.add_argument("--eval_llm", type=str, default='gpt3.5')
    parser.add_argument("--qa", type=int, default=5, help="qa num")
    parser.add_argument("--iter", type=int, default=5, help="evaluation")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument('--master_port', type=str, default="28900")

    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    args = parser.parse_args()
    if args.patch_size == 8:
        args.pretrained_weights = './dino_checkpoint/dino_vitbase8_pretrain_full_checkpoint.pth'
    elif args.patch_size == 16:
        args.pretrained_weights = './dino_checkpoint/dino_vitbase16_pretrain_full_checkpoint.pth'
    
    return args

def main():
    args = args_parse()
    eval_img_path = args.eval_file
    
    # Prepare an input for the model
    img_paths, prompt_lists, described_lists, key_words = [], [], [], []

    img_paths = read_into_list(eval_img_path, img_paths)
    if args.eval_type == 'rep':
        if 'woqa' in args.eval_file:
            file = 'woqa'
        else:
            file='orig'
        result_file = './'+args.eval_llm+file+'automatic_eval_rep.txt'
        vitb16 = load_model_dino(args)
        transform = pth_transforms.Compose([
            pth_transforms.Resize(256),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    elif args.eval_type == 'qa':
        if 'woqa' in args.eval_file:
            file = 'woqa'
        else:
            file='orig'
        result_file = './'+args.eval_llm+file+'automatic_eval_qa'+str(args.qa)+'_iter'+str(args.iter)+'.txt'
        if 'gpt' in args.vlm_model:
            args.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
    art_score, place_score, logo_score, char_score, product_score = 0,0,0,0,0 
    art_cnt, place_cnt, logo_cnt, char_cnt, product_cnt = 0,0,0,0,0
    art_remove, pl_remove, lo_remove, ch_remove, pr_remove = 0,0,0,0,0
    for i in range(len(img_paths)):
        img_paths[i] = img_paths[i].replace('\t', ' ')
        target_img_path = img_paths[i].split(', ')[0]
        gen_img_path = img_paths[i].split(', ')[1]
        target_img = Image.open(target_img_path)
        gen_img = Image.open(gen_img_path)
        
        if args.eval_type == 'qa':
            q_list, a_list = qa_gen_eval(args, target_img, target_img_path)
            respons_list1 = QA_score_ans(args, target_img, target_img_path, q_list, a_list)
            respons_list2 = QA_score_ans(args, gen_img, gen_img_path, q_list, a_list)
            score_target, score_gen = 0, 0
            for _ in range(args.iter):
                score_target += QA_eval_llm(args, q_list, a_list, respons_list1)
                score_gen += QA_eval_llm(args, q_list, a_list, respons_list2)

                print(score_target, score_gen)
            print(img_paths[i], score_target/args.iter, score_gen/args.iter)

            if 'Art' in img_paths[i]:
                art_score += score_gen/args.iter
                art_cnt += 1
            elif 'Ownership' in img_paths[i]:
                place_score += score_gen/args.iter
                place_cnt += 1
    
            elif 'Logo' in img_paths[i]:
                logo_score += score_gen/args.iter
                logo_cnt += 1
    
            elif 'Character' in img_paths[i]:
                char_score += score_gen/args.iter
                char_cnt += 1
    
            elif 'Product' in img_paths[i]:
                product_score += score_gen/args.iter
                product_cnt +=1 

            with open(result_file, 'a') as file:
                if i ==0:
                    file.write(f"evaluating with QA {args.qa}\n")
                file.write(f"{img_paths[i].split(', ')[0].split('/')[5:]}, {img_paths[i].split(', ')[1]}: {score_target/args.iter}, {score_gen/args.iter}, {q_list}, {a_list}, {respons_list1}, {respons_list2}\n")

        if args.eval_type == 'rep':
            target_img = transform(target_img)
            gen_img = transform(gen_img)

            gen_feat = extract_feat(gen_img.unsqueeze(0), vitb16)
            target_feat = extract_feat(target_img.unsqueeze(0), vitb16)

            score = similarity_score(gen_feat, target_feat)
            print(img_paths[i], score)
            if 'Art' in img_paths[i]:
                art_score += score.item()
                art_cnt += 1
                if score.item()>85.00:
                    art_remove += 1
            elif 'Ownership' in img_paths[i]:
                place_score += score.item()
                place_cnt += 1
                if score.item()>85.00:
                    pl_remove += 1
    
            elif 'Logo' in img_paths[i]:
                logo_score += score.item()
                logo_cnt += 1
                if score.item()>85.00:
                    lo_remove += 1
    
            elif 'Character' in img_paths[i]:
                char_score += score.item()
                char_cnt += 1
                if score.item()>85.00:
                    ch_remove += 1
    
            elif 'Product' in img_paths[i]:
                product_score += score.item()
                product_cnt +=1 
                if score.item()>85.00:
                    pr_remove += 1
    
            with open(result_file, 'a') as file:
                if i ==0:
                    file.write(f"evaluating with B{args.patch_size}\n")
                file.write(f"{img_paths[i].split(', ')[0].split('/')[5:]} - {img_paths[i].split(', ')[1]}: {score}\n")
    print(art_score/art_cnt, place_score/place_cnt, logo_score/logo_cnt, char_score/char_cnt, product_score/product_cnt)
    print(art_cnt,  place_cnt,  logo_cnt,  char_cnt,  product_cnt)
    print( art_remove,  pl_remove,  lo_remove,  ch_remove,  pr_remove)
    with open(result_file, 'a') as file:
        file.write(f"art, place, logo, character, product\n")
        file.write(f"{art_score/art_cnt}, {place_score/place_cnt}, {logo_score/logo_cnt}, {char_score/char_cnt}, {product_score/product_cnt}\n")
        file.write(f"{art_cnt}, {place_cnt}, {logo_cnt}, {char_cnt}, {product_cnt}\n")
        file.write(f"{art_remove}, {pl_remove}, {lo_remove}, {ch_remove}, {pr_remove}\n")
        
if __name__ == "__main__":
    main()
