import json
import numpy as np
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--base_file',default="Data/M3CoT_VQA_llava_QCM_A_accuracy.jsonl",type=str)
parser.add_argument('--exp_file',default="Data/m3cot_gpt3.5_3_QCM_A_accuracy.jsonl",type=str)
args = parser.parse_args()

base_rows=[]
base_ids=[]
base_path=args.base_file
with open(base_path,'r') as f:
    for line in f:
        row=json.loads(line)
        base_rows.append(row)
        base_ids.append(row['id'])
base_rows=sorted(base_rows,key=lambda x: x['id'])

rows=[]
experimental_path=args.exp_file
with open(experimental_path,'r') as f:
    for line in f:
        row=json.loads(line)
        rows.append(row)
exp_rows=sorted(rows,key=lambda x: x['qid'])


difference_score=[]
for exp in exp_rows:
    if exp['qid'] not in base_ids:
        print("Error: different question")
        print(exp['qid'])
    score=exp['accuracy']-base_rows[base_ids.index(exp['qid'])]['accuracy']
    difference_score.append(score)
    
sample=len(difference_score)
n=1000
better=0
not_better=0
for i in range(n):
    sampling=np.random.choice(difference_score,sample)
    if np.sum(sampling)>0:
        better+=1
    else:
        not_better+=1
        
p_value=not_better/n
print(p_value)
