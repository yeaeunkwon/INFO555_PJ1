from datasets import load_dataset
from openai import OpenAI
import argparse
import json
import os
import random
parser=argparse.ArgumentParser()

parser.add_argument('--model',default="gpt3.5",type=str)
parser.add_argument('--shot',default=0, type=int)
parser.add_argument('--seed',default=42, type=int)
parser.add_argument('--cot',default=False,type=bool)
parser.add_argument('--llava_dataset',default='v2',type=str)
parser.add_argument('--format',default="QCM_A",type=str)
parser.add_argument('--output_dir',default="Data/GPT3.5",type=str)
parser.add_argument('--data_path',default="Data/M3CoT_img_caption_llava.jsonl",type=str)
args = parser.parse_args()

def get_instruction():
    if args.format =="QCM_A":
        instruction="""
        You are provided with a question and an image description, which is the context of the question. 
        Your task is to give the correct answer to the following question by selecting one of the given options.
            Follow the specific JSON format provided below:
            {"answer": <your answer>}
            Be sure to fill "<your answer>" with the one of the given options and the correct answer. 
            Utilize the relevant information from the description and avoid any unrelated explanations.
            """
    elif args.format=="QCM_AR":
        instruction="""
        You are provided with a question and an image description, which is the context of the question. 
        Your task is to give the correct answer to the questions by selecting one of the given options and explain why the answer is correct by analyzing the given image description and applying relevant knowledge. 
            Ensure your explanation stays concise and to the point, and the answer should be one of the given options. Follow the specific JSON format provided below:
            {"answer": "<your answer>",
                "rationale": "<your rationale>"}
            Fill "<your answer>" and "<your rationale>" with one of the given options, the answer and the clear rationale supporting the correctness of the answer. 
            Utilize the relevant information from the description and avoid any unrelated explanations.
        """
    
    return instruction

def get_choices(choices):
    options=["A","B","C","D","E"]
    option_prompt=' '.join([f"({options[i]}) {choice}" for i, choice in enumerate(choices)])
    return option_prompt

def get_prompt(question,image_captions,choices,cot=False):
    choices=get_choices(choices)
    zero_shot=""
    image_captions=f"Context: {image_captions}"
    
    if cot and args.shot==0:
        zero_shot="Let's think step by step."
        input_prompt=f"Question: {question}\n{image_captions}\n{choices}\n{zero_shot}\n"   
        return input_prompt
       
    prompt=f"Question: {question}\n{image_captions}\n{choices}\n"
    
    return prompt

def open_json(path):
    data=[]
    with open(path) as f:
        for line in f:
            row=json.loads(line)
            row['caption']=row['caption'].split("[/INST] ")[1]
            data.append(row)
    return data

def save_json(row):
    if args.cot:
        cot_label="cot"
    else:
        cot_label=""
    path=os.path.join(args.output_dir,f"m3cot_{args.model}_standard_{args.shot}{cot_label}_{args.format}_{args.llava_dataset}_json.jsonl")
    
    if os.path.exists(args.output_dir)==False:
        os.makedirs(args.output_dir)
        
    with open(path,'a') as f:
            f.write(json.dumps(row)+'\n')
            
def get_examples(data):
    
    prompt=""
    idxs=[254,73,3051]
    for idx in idxs[:args.shot]:
        
        prompt+=get_prompt(data[idx]['question'],data[idx]['caption'],data[idx]['choices'],args.cot)
        if args.format=="QCM_A":
            prompt+=f"Answer: ({data[idx]['answer']})\n"
        elif args.format=="QCM_AR":
            prompt+=f"Answer: {data[idx]['answer']}\nRationale: {data[idx]['rationlae']}\n"
                   
    return prompt
    
        

if __name__=='__main__':
    
    ds=open_json(args.data_path)
    
    seed=args.seed
    
    if args.shot>0:
        example_prompt=get_examples(ds)
    random.seed(seed)    
    random.shuffle(ds)
    example_list=['physics-1294','physics-1519','mathematics-597']
    API_KEY= MY_KEY
    client=OpenAI(api_key=API_KEY)
    for i,row in enumerate(ds):
        if row['id'] in example_list:
            continue
        question=row['question']
        options=row['choices']
        answer=f"({row['answer']})"
        caption=row['caption']
        prompt=get_prompt(question,caption,options,args.cot)
        instruction=get_instruction()
        
        if args.shot>0:
            prompt=f"{example_prompt}\n{prompt}"
            
        prompt=f"{instruction}\n{prompt}"     
        ### START OF CODE FROM EXTERNAL SOURCE (URL: https://platform.openai.com/docs/guides/text-generation/building-prompts)      
        response=client.chat.completions.create(model="gpt-3.5-turbo", 
                                                messages=[{"role":"user","content":prompt}],
                                                max_tokens=512,
                                                temperature=0.01,
                                                top_p=1.0,
                                                frequency_penalty=0.0,
                                                presence_penalty=0.0)

        pred=response.choices[0].message.content
        
        ### END OF CODE FROM EXTERNAL SOURCE (URL: ...)
        pred={'qid':row['id'],
            'question':question,
            'options':options,
            'gold_answer':answer,
            'prediction':pred,
            'domain':row['domain']
            }
        if args.cot:
            pred['actual_rationale']=row['rationlae']
        save_json(pred)
        
