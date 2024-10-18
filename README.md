# INFO555_PJ1
This repository was built for the submission of the first project in INFO555 class (FALL2024).

## LLMs is Blind: LLMs can answer the Visual-Question-Answering without seeing the image.
### Problem
This project investigates the text-only LLM(GPT-3.5-turbo) on VQA task with image description instead of the orginal images.
### Dataset
M3CoT, which is multi-modal and multi-domain dataset, from HuggingFace libraray is used for the VQA task. 
Dataset : https://huggingface.co/datasets/LightChen2333/M3CoT

### Generated Data
M3CoT_img_caption_llava.jsonl
mathvista_img_caption_llava_v2.jsonl
M3CoT_VQA_llava_QCM_A.jsonl 
m3cot_gpt3.5_standard_0_QCM_A_v2_json.jsonl

m3cot_gpt3.5_standard_0cot_QCM_AR_v2_json.jsonl

m3cot_gpt3.5_standard_1_QCM_A_v2_json.jsonl

m3cot_gpt3.5_standard_1cot_QCM_AR_v2_json.jsonl

m3cot_gpt3.5_standard_2_QCM_A_v2_json.jsonl

m3cot_gpt3.5_standard_2cot_QCM_AR_v2_json.jsonl

m3cot_gpt3.5_standard_3_QCM_A_v2_json.jsonl

m3cot_gpt3.5_standard_3cot_QCM_AR_v2_json.jsonl
Gpt_3.5.py

Prompting_evaluation.py

finetuning_evaluation.py

finetuning_m3cot.py

mathvista_test.py

run_llava.py

