import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

import clip
import torch.nn as nn

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize                                


def gen_qa_list(n=3):
    # if type == 'question':
    if n == 3:
        number = "three"
    elif n == 5:
        number = "five"
    elif n==10:
        number = "ten"

    system_prompt = f"You are an expert question-answer generation based on the given image. \
        You takes a image as input and generate question and answer pairs about the image as output."
    user_prompt = f"Your task is to generate {number} question and its' respective answer pairs based on the given image. Generate question about object, size, shape, color or overall context and also generate answer for each questions. \
        Respond with each question in between <QUESTION> and </QUESTION> and respond with each concise answer in between <ANSWER> and </ANSWER>, eg: \
    1. <QUESTION>Question 1</QUESTION>\n \
    2. <QUESTION>Question 2</QUESTION>\n \
    3. <QUESTION>Question 3</QUESTION>\n"
    if n>3:
        for i in range(4, n):
            user_prompt += f"{n}. <QUESTION>Question {n}</QUESTION>\n"
    
    return system_prompt, user_prompt

def qa_parse(text):
    """
    1. <QUESTION> What is this image?</QUESTION>\n \
    <ANSWER>Answer 1</ANSWER>\n \
    2. <QUESTION> Question 2</QUESTION>\n \
    <ANSWER>Answer 2</ANSWER>\n \
    3. <QUESTION> Question 3</QUESTION>\n \
    <ANSWER>Answer 3</ANSWER>\n"
    """
    q_text_list = text.split('<QUESTION>')
    a_text_list = text.split('<ANSWER>')
    Q_list, A_list = [], []
    
    for n in range(len(q_text_list)):
        if '</QUESTION>' in q_text_list[n]:
            Q_list.append(q_text_list[n].split('</QUESTION>')[0])
    for n in range(len(a_text_list)):
        if '</ANSWER>' in a_text_list[n]:
            A_list.append(a_text_list[n].split('</ANSWER>')[0])
    
    return Q_list, A_list    

def remove_unwanted(texts):
    texts = texts.replace("<", "")
    texts = texts.replace(">", "")
    texts = texts.replace("/", "")
    texts = texts.replace(":", "")
    texts = texts.replace("INS", "")
    texts = texts.replace("PROMT", "")
    texts = texts.replace(",,", ",")
    return texts

def log_folder(args):
    folder_name = args.seed_prompt_model+'_'+args.model+'_'+args.generation_blackbox+'/'+\
        'S'+str(args.seed_automated)+str(args.seed_inst_optim_llm)+'_'+str(args.seed_automated_strategy)+'_n'+str(args.unchange_update_num)+\
        'A'+str(args.model)+'_'+str(args.rephrase_automated_strategy)+'_n'+str(args.rephrase_iter)+str(args.file_suffix)+'/'
    return folder_name

def log_in_text(args, output_type, text):
    if args.logging:
        folder_name = log_folder(args)
        full_folder = "./log/"+folder_name
        os.makedirs(full_folder, exist_ok=True)
        file_name = full_folder+"tracking.txt"
        text_total = f"Stage-{output_type}: {text}\n"
        if args.local_rank == 0:
            with open(file_name, 'a') as file:
                file.write(text_total)
    
def similarity_score(feat1, feat2):
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    if feat2.shape[0] == feat1.shape[0]:
        score = cos(feat1, feat2).mean()
    else:
        feat1 = feat1.repeat(feat2.shape[0], 1)
        score = cos(feat1, feat2)
        score = score.mean()

    score = (score +1.0)*50
    return score

def clip_extract(img=None, text=None, transform_need=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model, preprocess = clip.load('ViT-B/32', device)
    transform_img = Compose([
            Resize(clip_model.visual.input_resolution, interpolation=BICUBIC),
            CenterCrop(clip_model.visual.input_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    if not (img is None):
        if transform_need:
            img = transform_img(img.reshape(-1, 3, 512, 512)).to(device)
        else:
            img = preprocess(img).unsqueeze(0).to(device)
    if not (text is None):
        text_list = text
        for i in range(len(text)):
            text_list = text[i].split('.')    
        text = clip.tokenize(text_list).to(device)    

    with torch.no_grad():
        if not (img is None):
            image_features = clip_model.encode_image(img)
        if not (text is None):
            text_features = clip_model.encode_text(text)
    
    del clip_model
    torch.cuda.empty_cache()

    if img is None:
        return text_features
    elif text is None:
        return image_features
    else:
        return (image_features, text_features) 
    
def revise_template(model, system_prompt='', user_prompt=''):
    if model == 'llava':
        input_text = f"USER: <image>\n{user_prompt} ASSISTANT:"
    elif model == 'gpt3.5':
        input_text = {'system': system_prompt, 'user': user_prompt}
    elif model == 'gpt4-vision':
        input_text = {'system': system_prompt, 'user': user_prompt}
    else:
        input_text = user_prompt
    
    return input_text

def load_model(model_type, deepspeed_check=False, master_port=8888):
    # Load the model
    if model_type == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained("./llava-models/llava-1.5-7b-hf")
        if deepspeed_check:
            import deepspeed
            ## deepspeed inference
            if torch.cuda.device_count()<=1:
                deepspeed.init_distributed("nccl", distributed_port=master_port)

            ds_model = deepspeed.init_inference(
                        model, 
                        tensor_parallel={"tp_size": torch.cuda.device_count()},
                        dtype=torch.float16,
                        max_out_tokens=1024,  #big difference in generation: 2048+1024
                        injection_policy={LlavaForConditionalGeneration: ('self_attn.o_proj', 'mlp.down_proj')},
                )
        tokenizer = AutoProcessor.from_pretrained("./llava-models/llava-1.5-7b-hf")
    elif model_type == 'gpt4-vision' or model_type == 'gpt3.5':
        model, tokenizer = None, None
    else:
        raise NotImplementedError

    return model, tokenizer


def read_into_list(file_path, list_name):
    # Open the text file and read the paths
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and newline characters from each line
            stripped_line = line.strip()
            # Add the cleaned-up line (path) to the list
            list_name.append(stripped_line)
    return list_name

