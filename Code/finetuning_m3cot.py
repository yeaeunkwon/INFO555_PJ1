import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset,DataLoader
import json
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
import random
import os
import argparse
from finetuning_evaluation import calculate_accuracy

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
        
        inputs=self.tokenizer.batch_encode_plus([x],max_length=self.input_len,pad_to_max_length=True,return_tensors='pt',truncation=True)
        target=self.tokenizer.batch_encode_plus([y],max_length=self.exp_len,pad_to_max_length=True,return_tensors='pt',truncation=True)
        
        input_ids=inputs['input_ids'].squeeze()
        attention_mask=inputs['attention_mask'].squeeze()
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
        for i,data in enumerate(dataloader):
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
            
    return predictions,targets
    
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
            
def T5Train():
    
    seed=args.seed
    set_seed(seed)
    
   
    model_name="Flan-t5"
   
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", device_map="auto")
    
    dataset_path="Data/CoT_prompting/M3CoT_img_caption_llava.jsonl"
  
  
    data,label,domain=get_train_data(dataset_path,args.rationale)

    train_data,val_data,train_label,val_label=train_test_split(data,label,test_size=0.2,random_state=seed,shuffle=True,stratify=domain)
    
    training_set=QADataset(train_data,train_label,tokenizer,args.input_len,args.exp_len)
    val_set=QADataset(val_data,val_label,tokenizer,args.input_len,args.exp_len)

    training_loader=DataLoader(training_set,batch_size=args.batch_size, shuffle=True)
    val_loader=DataLoader(val_set,batch_size=args.batch_size, shuffle=False)
    
    optimizer=torch.optim.AdamW(params=model.parameters(),lr=args.lr)
    
    result_file=os.path.join(args.output_dir,"m3cot_training.txt")
    
    best_acc=0
    for epoch in range(args.epoch):
        model_train(epoch,tokenizer,model,device,training_loader,optimizer)
        

        predictions,gold=model_validate(epoch,tokenizer,model,device,val_loader,args.exp_len)
        print(len(predictions),len(gold))
       
        acc=calculate_accuracy(predictions,gold)
        
        
        save_dict={"model_state_dict":model.state_dict(),
                   "optimizer_state_dict":optimizer.state_dict()}
        
        
        with open(result_file,'a') as f:
                f.write(f"model: {model_name}, acc:{acc},current_epoch: {epoch}\n")
        
        if acc>best_acc:
            best_acc=acc
            
            best_model=save_dict
            torch.save(best_model,os.path.join(args.output_dir,f"{model_name}_{epoch}_{acc}_finetuning_m3cot.pt"))
            
        for p,g in zip(predictions,gold):
            row={'predictoin':p,
                    'gold':g}
            with open(os.path.join(args.output_dir,f"M3CoT_{model_name}_{epoch}_{acc}_valid_predictions.jsonl"),'a') as f:
                f.write(json.dumps(row)+'\n')
        
   
            
if __name__=='__main__':
    
     
    if os.path.exists(args.output_dir)==False:
        os.makedirs(args.output_dir)
        
    T5Train()