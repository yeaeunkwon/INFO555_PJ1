import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
import json
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import random
import os
import argparse
from datasets import load_dataset
from finetuning_m3cot import QADataset
parser=argparse.ArgumentParser()

parser.add_argument('--batch_size',default=2,type=int)
parser.add_argument('---lr',default=1e-4, type=float)
parser.add_argument('--epoch',default=20,type=int)
parser.add_argument('--output_dir',default="./T5output/",type=str)
parser.add_argument('--seed',default=42,type=int)
parser.add_argument('--rationale',default=False,type=bool)
parser.add_argument('--input_len',default=200,type=int)
parser.add_argument('--exp_len',default=150,type=int)
args = parser.parse_args()

device='cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        

def get_test_data(dataset, image_captions,options):
    
    input_list=[]
    for i in range(len(dataset['testmini']['question'])):
        question=dataset['testmini']['question'][i]
        choices=dataset['testmini']['choices'][i]
        answer=dataset['testmini']['answer'][i]
    
        if choices:
            option_prompt=f"Options: {' '.join([options[j]+choice for j,choice in enumerate(choices)])}"
            answer_prompt=f"({options[choices.index(answer)]} {dataset['testmini']['answer'][i]}"
        else:
            option_prompt=''
            answer_prompt=answer
        input=f"Question:{question}\nContext:{image_captions[i]['caption']}\n{option_prompt}"
        input_list.append(input)
    return input_list
            
def T5Test():
    
    seed=args.seed
    set_seed(seed)
    
    model_name="Flan-t5"
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", device_map="auto")
    
    CHECKPOINT_PATH="/home/labuser/Summarization/T5output/Flan-t5_19_0.512987012987013_rationale_finetuning_m3cot.pt"
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)['model_state_dict'])
    
    math_data = load_dataset("AI4Math/MathVista")
    test_img_caption="Data/mathvista_img_caption_llava_v2.jsonl"
    caption_text=[]
    with open(test_img_caption,'r') as f:
        for line in f:
            row=json.loads(line)
            row['caption']=row['caption'].split("[/INST] ")[1]
            caption_text.append(row)
            
    options=["A","B","C","D","E","F","G","H"]
    test_data=get_test_data(math_data,caption_text,options)
    test_label=math_data['testmini']['answer']
    test_set=QADataset(test_data,test_label,tokenizer,args.input_len,args.exp_len)
    test_loader=DataLoader(test_set,batch_size=args.batch_size, shuffle=False)
    
    
    model.eval()
    predictions=[]
    targets=[]

    with torch.no_grad():
        for i,data in enumerate(test_loader):

            input_ids=data['input_ids'].squeeze().to(device,dtype=torch.long)
            attention_mask=data['attention_mask'].squeeze().to(device,dtype=torch.long)
            true_labels=data['target_ids'].to(device,dtype=torch.long)
            
            generated_ids=model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.exp_len,
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=2.0,
                early_stopping=True
            )
            
            if i%100==0:
                print(i)
            preds=[tokenizer.decode(id,skip_special_tokens=True,clean_up_tokenization_spaces=True) for id in generated_ids]
            decoded_labels=[tokenizer.decode(label,skip_special_tokens=True,clean_up_tokenization_spaces=True) for label in true_labels]    
            predictions.extend(preds)
            targets.extend(decoded_labels)

        
        
    for p,g,id,c in zip(predictions,targets,math_data['testmini']['pid'],math_data['testmini']['choices']):
        if c==None:
            continue
    
        row={'qid': id,
            'prediction':p,
            'gold':options[c.index(g)],
            'choices':c}
        with open(os.path.join(args.output_dir,f"Mathvista_{model_name}_QCM_A_test_predictions_v3.jsonl"),'a') as f:
            f.write(json.dumps(row)+'\n')
        
    
            
if __name__=='__main__':
    
     
    if os.path.exists(args.output_dir)==False:
        os.makedirs(args.output_dir)
        
    T5Test()