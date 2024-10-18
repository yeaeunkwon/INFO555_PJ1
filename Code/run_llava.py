from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch
import json
import argparse
import datasets

dataset = datasets.load_dataset("LightChen2333/M3CoT")

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",torch_dtype=torch.float16,low_cpu_mem_usage=True,load_in_4bit=True).to("cuda:0")
j=0
for i,data in enumerate(dataset['train']):
    
    if data['image']==None:
        continue
    ### START OF CODE FROM EXTERNAL SOURCE (URL: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text","text": "Describe the image in details. If there are numbers and letters in the image, read them correctly"},
                ],
        },
    ]
    
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=data['image'], text=text_prompt, return_tensors="pt",).to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=100)
    caption=processor.decode(output[0], skip_special_tokens=True)
    ### END OF CODE FROM EXTERNAL SOURCE (URL: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
    row={'id':data['id'],
            'question':data['question'],
            'choices':data['choices'],
            'caption':caption,
            'answer':data['answer'],
            'rationlae':data['rationale'],
            'domain':data['domain']}
   
    with open(f"M3CoT_img_caption_llava.jsonl",'a') as f:
        f.write(json.dumps(row)+'\n')