## LLMs is Blind: LLMs can answer the Visual-Question-Answering without seeing the image.
Vision for humans is another significant chan-
nel for processing information. However, hu-
man cognitive ability can visualize an image
based solely on descriptive text, using prior ex-
periences and accumulated knowledge. This
project investigates the ability of a text-only
LLM in a VQA task when provided the im-
age descriptions, instead of the original im-
age, with QA pairs, compared to that of a
multi-modal LLM. The standard and Chain-
of-Thought(CoT) prompts are given to Gpt-3.5-
turbo, and the image descriptions are obtained
from LLaVa. The results illustrate that CoT
strategy can enhance and surpass the multi-
modal LLM in a VQA task even though the
images are not directly provided to the text-
only LLM. The experiments were conducted
on the M3CoT dataset, addressing multiple do-
mains such as mathematics, science, and com-
monsense.
### Paper
I couldn't upload the paper in this repository because of the error that says "Unable to render code block". So, I uploaded the paper to my Drive, and this is the link to the [file](https://drive.google.com/file/d/195ztptTjsfzqGNOEEseaiaTU_CCnlfIF/view?usp=sharing) : https://drive.google.com/file/d/195ztptTjsfzqGNOEEseaiaTU_CCnlfIF/view?usp=sharing
### Problem
This project investigates the text-only LLM(GPT-3.5-turbo) on VQA task with image description instead of the orginal images.
![system](https://github.com/user-attachments/assets/9bfea4bd-c6df-4186-bfb9-979f9309e728)
### Dataset
M3CoT, which is multi-modal and multi-domain dataset, from HuggingFace libraray is used for the VQA task. [Dataset](https://huggingface.co/datasets/LightChen2333/M3CoT)

### Generated Data(Data Directory)
* M3CoT_img_caption_llava.jsonl : The image description data after inputting the images from M3CoT to [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
* mathvista_img_caption_llava_v2.jsonl :  The image description data after inputting the images from MathVista to [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
* M3CoT_VQA_llava_QCM_A.jsonl : The results of [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) on M3CoT Dataset.
* m3cot_gpt3.5_standard_0_QCM_A_v2_json.jsonl : the result of zero-shot prompting to GPT-3.5-turbo
* m3cot_gpt3.5_standard_1_QCM_A_v2_json.jsonl : the result of one-shot prompting to GPT-3.5-turbo
* m3cot_gpt3.5_standard_2_QCM_A_v2_json.jsonl : the result of two-shot prompting to GPT-3.5-turbo
* m3cot_gpt3.5_standard_3_QCM_A_v2_json.jsonl : the result of three-shot prompting to GPT-3.5-turbo
* m3cot_gpt3.5_standard_0cot_QCM_AR_v2_json.jsonl : the result of zero-shot prompting with CoT to GPT-3.5-turbo
* m3cot_gpt3.5_standard_1cot_QCM_AR_v2_json.jsonl : the result of one-shot prompting with CoT to GPT-3.5-turbo
* m3cot_gpt3.5_standard_2cot_QCM_AR_v2_json.jsonl : the result of two-shot prompting with CoT to GPT-3.5-turbo
* m3cot_gpt3.5_standard_3cot_QCM_AR_v2_json.jsonl : the result of three-shot prompting with CoT to GPT-3.5-turbo

### Code(Code Directory)
* Gpt_3.5.py : runnning GPT-3.5-turbo when giving the prompts
* Prompting_evaluation.py : Calculating the accuracy, BERTScore, and ROUGE after prompting.
* run_llava.py :  Obtaining the image descriptions from [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
* finetuning_m3cot.py : Fine-tuning the FLAN-T5-SMALL on the M3CoT dataset
* finetuning_evaluation.py : Calculating the accuracy of the result from the fine-tuning.
* mathvista_test.py : Testing the fine-tuned flan-t5-small model on [MathVista Dataset](https://huggingface.co/datasets/AI4Math/MathVista)


