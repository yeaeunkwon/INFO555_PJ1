from sklearn.metrics import accuracy_score 
import argparse
import json
import re
from rouge import Rouge



def calculate_accuracy(prediction,gold):
    acc=0
    for g,p in zip(gold,prediction):
        
        pattern = r'\(([A-Z])\)'
        g_choice=g.replace("Answer: ","").strip("()")
        
        matches = re.search(pattern, p)
        if matches:
            p_choice=matches.group().strip('()')
            
            if p_choice==g_choice:
                acc+=1
            
            
    return acc/len(prediction)

def test_accuracy(prediction,gold,choice):
    options=["A","B","C","D","E","F","G","H"]
    pattern = r'\(([A-Z])\)'
    matches = re.search(pattern,prediction)
    try:
        g_choice=options[choice.index(gold)]
    except:
        return 0
    if matches:
            p_choice=matches.group().strip('()')
            if g_choice==p_choice:
                return 1
    return 0


if __name__=="__main__": 
    path="Data/T5output/Mathvista_flan-t5_QCM_A_test_predictions_v2.jsonl"
    data=[]
    with open(path,'r') as f:
        for line in f:
            row=json.loads(line)
            data.append(row)
    acc=0
    num=0        
    for d in data:
        if d['choices']==None:
            continue
        acc+=test_accuracy(d['predictoin'],d['gold'],d['choices'])
        num+=1
        
            
