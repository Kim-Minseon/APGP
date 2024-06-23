import torch
import os 
from utils import remove_unwanted

def inference_llm(args, model, tokenizer, orig_text, image=None, model_type='gpt4-vision', N=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if N is None:
        N = args.num_rephrase
    
    if model_type == 'llava':
        input_ids = tokenizer(text=orig_text, images=image, return_tensors="pt")
        if args.deepspeed:
            input_ids = input_ids.to(device)
    elif model_type == 'gpt4-vision':
        # OpenAI API Key
        from openai import OpenAI
        if args.local_rank == 0:
            client = OpenAI()
            #organization='org-zoFcme4rM5EDHooSJGyJCb54',
            if image is None:
                    response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": orig_text['system']},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": orig_text['user']},
                            ],
                        }
                    ],
                    n = 1,
                    max_tokens=896,
                )
            else:
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": orig_text['system']},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": orig_text['user']},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image}"
                                    },
                                },
                            ],
                        }
                    ],
                    n = 1,
                    max_tokens=896,
                )
            output_seq = [response]
        else:
            output_seq = ['_']
        #torch.distributed.broadcast_object_list(output_seq_text, args.local_rank)
        output = [None for _ in range(torch.cuda.device_count())]
        torch.distributed.all_gather_object(output, output_seq[0])
        output_sequences = output[0]
    elif model_type == 'gpt3.5':
        from openai import OpenAI
        
        if args.local_rank == 0:
            client = OpenAI()

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": orig_text['system']},
                    {"role": "user", "content": orig_text['user']}
                ],
                n = 1,
                max_tokens=896,
            )
            output_seq = [response]
        else:
            output_seq = ['_']
        #torch.distributed.broadcast_object_list(output_seq_text, args.local_rank)
        output = [None for _ in range(torch.cuda.device_count())]
        torch.distributed.all_gather_object(output, output_seq[0])
        output_sequences = output[0]
    else:
        input_ids = tokenizer.encode(orig_text, return_tensors="pt")

    if model_type == 'llava':
        # Generate output
        model = model.to(device)
        output_sequences = []
        #if args.local_rank == 0:
        for _ in range(N):
            sequences = model.generate(**input_ids, 
                                        top_k=10,
                                        do_sample=True,
                                        max_length=2048)
            output_sequences.append(sequences)
        output_seq = output_sequences

        input_ids = input_ids.to('cpu')

    if model_type == 'llava':
         # Decode the generated text
        output_seq_text = list()
        for seqs in output_sequences:
            texts = tokenizer.batch_decode(seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            texts = texts.split('ASSISTANT:')[-1].replace('\n', '')
            output_seq_text.append(texts)
    elif model_type == 'gpt4-vision' or model_type == 'gpt3.5':
        output_seq_text = list()
        for seqs in output_sequences.choices:
            texts = seqs.message.content
            if 'INS' in texts:
                texts = texts.split('<INS>')[-1].split('</INS>')[0]
            if 'PROMPT' in texts:
                texts = texts.split('<PROMPT>')[-1].split('</PROMPT>')[0]
            
            output_seq_text.append(texts)
    return output_seq_text

