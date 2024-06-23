# Automatic Jailbreaking of the Text-to-Image Generative AI Systems (APGP)

Official PyTorch implementation of "Automatic Jailbreaking of the Text-to-Image Generative AI Systems".

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Citation](#citation)

## About
### TL;DR
> Commercial text-to-image systems (ChatGPT, Copilot, and Gemini) block copyrighted content to prevent infringement, but these safeguards can be easily bypassed by our automated prompt generation pipeline.

### Paper link & Project page
- Paper link: [Paper](https://arxiv.org/abs/2405.16567)
- Project page: [Project_Page](https://kim-minseon.github.io/APGP)

### Abstract
Recent AI systems have shown extremely powerful performance, even surpassing human performance, on various tasks such as information retrieval, language generation, and image generation based on large language models (LLMs). At the same time, there are diverse safety risks that can cause the generation of malicious contents by circumventing the alignment in LLMs, which are often referred to as jailbreaking. However, most of the previous works only focused on the text-based jailbreaking in LLMs, and the jailbreaking of the text-to-image (T2I) generation system has been relatively overlooked. In this paper, we first evaluate the safety of the commercial T2I generation systems, such as ChatGPT, Copilot, and Gemini, on copyright infringement with naive prompts. From this empirical study, we find that Copilot and Gemini block only 12% and 17% of the attacks with naive prompts, respectively, while ChatGPT blocks 84% of them. Then, we further propose a stronger automated jailbreaking pipeline for T2I generation systems, which produces prompts that bypass their safety guards. Our automated jailbreaking framework leverages an LLM optimizer to generate prompts to maximize degree of violation from the generated images without any weight updates or gradient computation. Surprisingly, our simple yet effective approach successfully jailbreaks the ChatGPT with 11.0% block rate, making it generate copyrighted contents in 76% of the time.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

List the prerequisites required to run this project. Include any necessary software, tools, or packages.

- Python >= 3.10
- Pytorch == 2.1.2
- deepspeed == 0.13.2
- cuda 12.1

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kim-minseon/APGP.git
    ```

2. Navigate to the project directory:
    ```bash
    cd APGP
    ```

3. Create a virtual environment:
    ```bash
    conda create --name copyright
    conda activate copyright
    ```

4. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Prepare data

Download your data and revise the dataset path in `/Dataset/img_path_*.txt` <br>
State the keyword for the target image in `/Dataset/keywords_*.txt`

### Generate high risk prompt

Command
```bash
sh run.sh $GPU_num $Master_port $VLM_in_seed_stage $LLM_for_seed_optim $LLM_for_revise_optim $T2I_model $seed_update_num $revise_update_num $save_file_name $data_path_name*
```

Example
```bash
sh run.sh 1 8888 gpt4-vision gpt3.5 gpt3.5 dalle3 3 5 all all
```

You can find your high risk prompt in `./results/*/*/score_keyword.txt` <br>
Try them in commercial T2I systems!

## Citation
> @article{kim2024automatic, <br>
  title={Automatic Jailbreaking of the Text-to-Image Generative AI Systems}, <br>
  author={Kim, Minseon and Lee, Hyomin and Gong, Boqing and Zhang, Huishuai and Hwang, Sung Ju}, <br>
  journal={ICML 2024 Workshop NextGenAISafety}, <br>
  year={2024} <br>
} <br>
