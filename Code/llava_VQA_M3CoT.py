import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import json
import datasets

def open_json(path):
    data=[]
    with open(path) as f:
        for line in f:
            row=json.loads(line)
            row['caption']=row['caption'].split("[/INST] ")[1]
            data.append(row)
    return data

def get_choices(choices):
    options=["A","B","C","D","E"]
    option_prompt=' '.join([f"({options[i]}) {choice}" for i, choice in enumerate(choices)])
    return option_prompt



dataset_path="Data/M3CoT_img_caption_llava.jsonl"
dataset = datasets.load_dataset("LightChen2333/M3CoT")

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",torch_dtype=torch.float16,low_cpu_mem_usage=True,load_in_4bit=True).to("cuda:0")



captions=open_json(dataset_path)
ids=[caption['id'] for caption in captions]
print(len(ids))
train_data=dataset['train']

for i,data in enumerate(train_data):
    if data['id'] not in ids:
        continue
   
    options=get_choices(data['choices'])


    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text","text": """
    You are provided with a question and an image description, which is the context of the question. 
        Your task is to give the correct answer to the questions by selecting one of the given options and explain why the answer is correct by analyzing the given image description and applying relevant knowledge. 
            Ensure your explanation stays concise and to the point, and the answer should be one of the given options. Follow the specific JSON format provided below:
            {"answer": "<your answer>",
                "rationale": "<your rationale>"}
            Fill "<your answer>" and "<your rationale>" with one of the given options, the answer and the clear rationale supporting the correctness of the answer. 
            Utilize the relevant information from the description and avoid any unrelated explanations.\n
            """+f"Question: {data['question']}\nOptions: {options}"},
                    ],
            },
        ]
    ### START OF CODE FROM EXTERNAL SOURCE (URL: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=data['image'], text=text_prompt, return_tensors="pt",).to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=150)
    answer=processor.decode(output[0], skip_special_tokens=True)
    ### END OF CODE FROM EXTERNAL SOURCE (URL: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
    row={'id':data['id'],
             'question':data['question'],
             'choices':data['choices'],
             'prediction':answer.split("[/INST]")[1],
             'gold_answer':f"({data['answer']})",
             'rationlae':data['rationale'],
             'domain':data['domain']}
        
    print("\n"+text_prompt)
        
    print(answer)
    with open(f"Data/M3CoT_VQA_llava_QCM_A.jsonl",'a') as f:
        f.write(json.dumps(row)+'\n')