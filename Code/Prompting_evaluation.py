import argparse
import json
import re
from rouge import Rouge
from bert_score import score

parser=argparse.ArgumentParser()

parser.add_argument('--rationale',default=False,type=bool)
parser.add_argument('--path',default="Data/GPT3.5/m3cot_gpt3.5_standard_3cot_QCM_AR_v2_json.jsonl",type=str)
args = parser.parse_args()

def open_json(path):
    data=[]
    with open(path) as f:
        for line in f:
            row=json.loads(line)
            data.append(row)
                
    return data

def r_prompting_answer_accuracy(pred,gold,choices,rationale=False):
    options=["A","B","C","D","E"]
    pred=pred.strip(" ").strip("{}")
    if rationale:
        pred=pred.split("rationale")[0].strip(" ")
    pred=pred.replace('"answer":',"")  
    pred=pred.replace('"','')  
    pred=pred.replace("\t", "").replace("\n", "").replace(',','')
    
    #The generated answers have four types
    # (A)
    # A
    # A.
    # <the answer> instead of the option
    
    pattern = r'\(([A-Z])\)'
    option=re.search(pattern,pred)
    g_option=gold.strip('()')
    p_option=pred.strip('()')
    p_option=p_option.replace('.','')
    
    if p_option in options:
        pred=p_option
    else:
        pred=pred.strip(' ')
    
    acc=0
    if pred==g_option:
        acc=1
    elif pred not in options and pred in choices:
        if g_option==options[choices.index(pred)]:
            acc=1
    elif option:
        if g_option==option.group():
            acc=1
    else:
        acc=0
    return acc    

def rouge_score(pred,gold_rationale):
    pred=pred.strip(" ").strip("{}")
    pred=pred.split("rationale")[1][2:].strip('"')
    rouge=Rouge()
    scores=rouge.get_scores(gold_rationale,pred)
    
    return scores[0]['rouge-1']['f'],scores[0]['rouge-2']['f'],scores[0]['rouge-l']['f']

def bert_score(pred,gold_rationale):
    pred=pred.strip(" ").strip("{}")
    pred=pred.split("rationale")[1][2:].strip('"')
    p,r,f1=score([gold_rationale],[pred],lang="en")
    return p,r,f1
    
    
if __name__=='__main__':
    
    path=args.path
    outputs=open_json(path)
   
    acc=0
    domain_acc={"science":0,"commonsense":0,"mathematics":0}
    for output in outputs:
        o_acc=r_prompting_answer_accuracy(output['prediction'],output['gold_answer'],output['options'],args.rationale)
        acc+=o_acc
        if o_acc==1:
            domain_acc[output['domain']]+=1
    
   
    rouge_dict={"rouge1":0,"rouge2":0,"rougel":0}
    domain_rouge={"science_rouge1":0,"science_rouge2":0,"science_rougel":0,
                  "commonsense_rouge1":0,"commonsense_rouge2":0,"commonsense_rougel":0,
                  "mathematics_rouge1":0,"mathematics_rouge2":0,"mathematics_rougel":0}
    for output in outputs:
        r1,r2,rl=rouge_score(output['prediction'],output['actual_rationale'])
        rouge_dict['rouge1']+=r1
        rouge_dict['rouge2']+=r2
        rouge_dict['rougel']+=rl
        domain_rouge[f"{output['domain']}_rouge1"]+=r1
        domain_rouge[f"{output['domain']}_rouge2"]+=r2
        domain_rouge[f"{output['domain']}_rougel"]+=rl
   
    bert_dict={"p":0,"r":0,"f1":0}
    domain_bert={"science":0,"commonsense":0,"mathematics":0}
    for i,output in enumerate(outputs):
        p,r,f1=bert_score(output['prediction'],output['actual_rationale'])
        bert_dict['p']+=p
        bert_dict['r']+=r
        bert_dict['f1']+=f1
        domain_bert[output['domain']]+=f1
        
        if i%100==0:
            print(i)
   
  