import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer,AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
import wandb
import random
import os
import re
from bert_score import score
from sacrebleu.metrics import BLEU
from sacrebleu import corpus_chrf,corpus_bleu
from rouge import Rouge
import argparse
from datasets import load_dataset
from evaluation import calculate_accuracy

parser=argparse.ArgumentParser()

parser.add_argument('--batch_size',default=2,type=int)
parser.add_argument('---lr',default=1e-4, type=float)
parser.add_argument('--epoch',default=20,type=int)
parser.add_argument('--output_dir',default="./T5output/",type=str)
parser.add_argument('--seed',default=42,type=int)
parser.add_argument('--rationale',default=True,type=bool)
args = parser.parse_args()

device='cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

class QADataset(Dataset):
    
    def __init__(self,data,label,tokenizer,input_len,exp_len):
        self.tokenizer=tokenizer
        self.data=data
        self.label=label
        self.input_len=input_len
        self.exp_len=exp_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        x=self.data[idx]
        y=self.label[idx]
        
        source=self.tokenizer.batch_encode_plus([x],max_length=self.input_len,pad_to_max_length=True,return_tensors='pt',truncation=True)
        target=self.tokenizer.batch_encode_plus([y],max_length=self.exp_len,pad_to_max_length=True,return_tensors='pt',truncation=True)
        
        input_ids=source['input_ids'].squeeze()
        attention_mask=source['attention_mask'].squeeze()
        target_ids=target['input_ids'].squeeze()
        return {
            'input_ids':input_ids.to(dtype=torch.long),
            'attention_mask':attention_mask.to(dtype=torch.long),
            'target_ids':target_ids.to(dtype=torch.long),
            'gold_answer':[y]
        }
        

def model_train(epoch,tokenizer,model,device,dataloader,optimizer):
    
    model.train()
    
    for i,data in enumerate(dataloader,0):
        ids=data['input_ids'].to(device,dtype=torch.long)
        mask=data['attention_mask'].to(device,dtype=torch.long)
        y=data['target_ids'].to(device,dtype=torch.long)
        
        decoding_inputs=y[:,:-1].contiguous()
        decoding_labels=y[:,1:].clone().detach()
        decoding_labels[y[:,1:]==tokenizer.pad_token_id]=-100

        
        outputs=model(input_ids=ids,attention_mask=mask,decoder_input_ids=decoding_inputs,labels=decoding_labels)
        loss=outputs[0]
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()


        if i%100==0:
            print(f"Epoch : {epoch}, Loss : {loss.item()}")
      
        
def model_validate(epoch,tokenizer,model,device,dataloader,exp_len):
    
    model.eval()
    predictions=[]
    targets=[]

    with torch.no_grad():
        for i,data in enumerate(dataloader,0):
            ids=data['input_ids'].to(device,dtype=torch.long)
            mask=data['attention_mask'].to(device,dtype=torch.long)
            true_labels=data['target_ids'].to(device,dtype=torch.long)
            
            generated_ids=model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=exp_len,
                num_beams=2,
                repetition_penalty=2.0,
                length_penalty=1.0,
                early_stopping=True
            )
            preds=[tokenizer.decode(g,skip_special_tokens=True,clean_up_tokenization_spaces=True) for g in generated_ids]
            decoded_labels=[tokenizer.decode(t,skip_special_tokens=True,clean_up_tokenization_spaces=True) for t in true_labels]
            if i%100==0:
                print(f"{epoch} | {i}")
                
            predictions.extend(preds)
         
            targets.extend(decoded_labels)
    print(preds,targets)
            
    return predictions,targets

def model_test(tokenizer,model,device,test_data,input_len,exp_len):
    model.eval()
    predictions=[]
    targets=[]

    with torch.no_grad():
        for i,data in enumerate(test_data):
            
            source=tokenizer.batch_encode_plus(data,max_length=input_len,pad_to_max_length=True,return_tensors='pt',truncation=True)

            input_ids=source['input_ids'].squeeze().to(device,dtype=torch.long)
            attention_mask=source['attention_mask'].squeeze().to(device,dtype=torch.long)
            
            
            generated_ids=model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=exp_len,#원래 150
                num_beams=1,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds=[tokenizer.decode(g,skip_special_tokens=True,clean_up_tokenization_spaces=True) for g in generated_ids]
                
            predictions.extend(preds)
         
    return predictions
    
def get_train_data(path,rationale=False):
    input_list=[]
    output_list=[]
    options=["A","B","C","D","E"]
    domains=[]
    with open(path,'r') as f:
        for line in f:
            row=json.loads(line)
            caption=row['caption'].split("[/INST] ")[1]
            domains.append(row['domain'])
            option_prompt=' '.join([f"({options[i]}) {choice}" for i, choice in enumerate(row['choices'])])
            input_text=f"Question:{row['question']}\nContext: {caption}\Options: {option_prompt}"
            output_text=f"Answer: ({row['answer']})"
            if rationale:
                output_text+=f"\nRationale: {row['rationlae']}"
            input_list.append(input_text)
            output_list.append(output_text)
    return input_list,output_list,domains


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
            
def T5Train(params):
    
    seed=params['seed']
    set_seed(seed)
    
    publisher="google-t5/"#"allenai/"
    model_name="t5-base"#"unifiedqa-t5-base"#"google/flan-t5-small" #FLAN-T5
    #model_name="Flan-t5"
    tokenizer=T5Tokenizer.from_pretrained(publisher+model_name)
    model=T5ForConditionalGeneration.from_pretrained(publisher+model_name).to(device)
    #tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", device_map="auto")
    
    CHECKPOINT_PATH="/home/labuser/Summarization/T5output/t5-base_19_0.5243506493506493_finetuning_m3cot.pt"
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)['model_state_dict'])
    
    math_data = load_dataset("AI4Math/MathVista")
    test_img_caption="/home/labuser/Applied_NLP/CoT_prompting/mathvista_img_caption_llava_v2.jsonl"
    caption_text=[]
    with open(test_img_caption,'r') as f:
        for line in f:
            row=json.loads(line)
            row['caption']=row['caption'].split("[/INST] ")[1]
            caption_text.append(row)
            
    options=["A","B","C","D","E","F","G","H"]
    test_data=get_test_data(math_data,caption_text,options)
    test_label=math_data['testmini']['answer']
    test_set=QADataset(test_data,test_label,tokenizer,params['input_len'],params['exp_len'])
    test_loader=DataLoader(test_set,batch_size=params['batch_size'], shuffle=False)
    optimizer=torch.optim.AdamW(params=model.parameters(),lr=params['lr'])
    
    
    
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
                max_length=params['exp_len'],#원래 150
                num_beams=1,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            
            if i%100==0:
                print(i)
            preds=[tokenizer.decode(g,skip_special_tokens=True,clean_up_tokenization_spaces=True) for g in generated_ids]
            decoded_labels=[tokenizer.decode(t,skip_special_tokens=True,clean_up_tokenization_spaces=True) for t in true_labels]    
            predictions.extend(preds)
            targets.extend(decoded_labels)

        
        
    for p,g,id,c in zip(predictions,targets,math_data['testmini']['pid'],math_data['testmini']['choices']):
        if c==None:
            continue
        print(c)
        row={'qid': id,
            'prediction':p,
            'gold':options[c.index(g)],
            'choices':c}
        with open(os.path.join(params['output_dir'],f"Mathvista_{model_name}_QCM_A_test_predictions_v3.jsonl"),'a') as f:
            f.write(json.dumps(row)+'\n')
       
        
    
            
if __name__=='__main__':
    
     
    if os.path.exists(args.output_dir)==False:
        os.makedirs(args.output_dir)
        
    params={'batch_size':args.batch_size,
            'lr':args.lr,
            'output_dir':args.output_dir,
            'epoch':args.epoch,
            'seed':args.seed,
            'input_len':200,
            'exp_len':512,
            'rationale':args.rationale}
    T5Train(params)