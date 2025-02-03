import sys
import time
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

# Default model path
models_path = r".\models"
model_default = r"phi-4-q4"
model_split = r"phi-4-q4"

local_model_path= "./embedding_model/bge-m3"
# Default database path
persist_directory=r".\db"

# Prompt templates
'''
template_query = ("System: 現在擁有以下資料 {docs}\n\n" +
    "User: {question}\n\n請用正體中文回答問題\n\n" +
    "Assistant: 根據資料, "
    )

template_split = (
    "User: if there multiple commands or questions in the sentence \"{question}\"." + 
    "Rephrase it into multiple sentances in individual lines and surround them with semicolon in both beginning and end. " +
    "\n\nAssistant:"
    )

template_query = (
    "<|im_start|>system<|im_sep|>\n"+
    "現在擁有以下資料 \n\n{docs}\n\n" +
    "<|im_end|>\n\n"+
    "<|im_start|>user<|im_sep|>\n"+
    "{question}\n\n請用正體中文回答問題\n\n" +
    "<|im_end|>\n\n"+
    "<|im_start|>assistant<|im_sep|>\n"
    )
'''

#for phi 4
template_query = (
    "<|im_start|>system<|im_sep|>\n"+
    "According to the following document \n\n{docs}\n\n" +
    "Please answer user's questions by the same language that user use."+
    "<|im_end|>\n\n"+
    "<|im_start|>user<|im_sep|>\n"+
    "{question}\n\n" +
    "<|im_end|>\n\n"+
    "<|im_start|>assistant<|im_sep|>\n"
    )
template_split = (
    "<|im_start|>user<|im_sep|>\n"+
    "if there multiple commands or questions in the sentence \"{question}\"." +
    "Rephrase it into multiple sentances in individual lines and surround them with semicolon in both beginning and end. " +
    "<|im_end|>\n\n"+
    "<|im_start|>assistant<|im_sep|>\n"
    )
'''
template_query = (
    "<｜System｜>\n"+
    "According to the following document \n\n{docs}\n\n" +
    "Please answer user's questions by the same language that user use."+
    "<|im_end|>\n\n"+
    "<｜User｜>\n"+
    "{question}\n\n" +
    "<｜Assistant｜>\n"
    )
template_split = (
    "<|im_start|>user<|im_sep|>\n"+
    "if there multiple commands or questions in the sentence \"{question}\"." +
    "Rephrase it into multiple sentances in individual lines and surround them with semicolon in both beginning and end. " +
    "<|im_end|>\n\n"+
    "<|im_start|>assistant<|im_sep|>\n"
    )
'''

# Parameter for llm
gpu_id = 1
n_gpu_layers = 25  # Metal set to 1 is enough. 
n_ctx = 16384
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
max_tokens = n_ctx
temperature = 0.1
top_p = 0.95#0.1

embedding = HuggingFaceEmbeddings(model_name=local_model_path,model_kwargs = {'device': 'cuda:1'})#GPT4AllEmbeddings()#HuggingFaceEmbeddings(model_name=local_model_path,model_kwargs = {'device': 'cuda:1'})

# parameter for similarity test
k = 10 # number of document will be used
score_threshold = 0.5 #1 is higher the similarity, 0 is lower.

# String to return when there is no answer for the question
ans_null = "不好意思, 我無法回答這個問題"
#use "{question}" as the question provided

N = 3 #上限的問題數量

models = {}

def init():
    global model_default
    global model_split
    path_list = getFileList(models_path)
    for path in path_list:
        name = Path(path).stem
        #WARNING ISSUE: https://github.com/langflow-ai/langflow/issues/1820
        llm = LlamaCpp(
            tensor_split = [0,1],
            main_gpu = gpu_id,
            n_gpu_layers = n_gpu_layers,
            model_path = path,
            n_batch = n_batch,
            n_ctx = n_ctx,
            temperature = temperature, 
            max_tokens = max_tokens,
            top_p = top_p,
            f16_kv = True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            verbose = True, # May need to set to True, somehow it breaks when False...
        )
        models[name]=llm
        model_default = name
        model_split = name
        print("Model: '" + name + "' loaded")
        break #只讀取一個模型

def getQuestions(query, llm):
    llm = models[model_split]
    #Force to use llama..
    print("Original query: " + query)
    questions = []
    prompt_split = PromptTemplate.from_template(template_split)
    llm_chain_split = prompt_split | llm.bind(stop=["Assistant:"])
    qs = llm_chain_split.invoke({"question": query},{"recursion_limit": 1})
    print("First layer of model: "+qs)
    qlist = qs.split("\n")
    count = 0 
    for line in qlist:
        q = line.rstrip()
        if(len(q)<3):
            continue
        if(q.startswith(";") and q.endswith(";")):
            q=q[1:len(q)-1].strip()
            if (not q in questions):
                if (N==-1 or count<N):
                    questions.append(q)
                    count +=1
                else:
                    break
        
    return questions

def runQuery_NoLimit(query, persist_directory=persist_directory, model=model_default):
    #print("==============================Prompt=================================")
    prompt = PromptTemplate.from_template(template_query)
    llm = None
    if model in models.keys():
        llm = models[model]
    else:
        print("Model: '"+model+"' not found, using default model.")
        llm = models[model_default]
    #print("==============================CHAIN==================================")
    llm_chain = prompt | llm.bind(stop=["<|im_start|>user<|im_sep|>\n"])
    '''
    questions = getQuestions(query, llm)
    for question in questions:
        #question = r"Who is the director of Sue & Bill Gross Stem Cell Research Center?"
        docs = vectorstore.similarity_search_with_relevance_scores(
            question,
            k=k,
            score_threshold=score_threshold
        )
        if(len(docs)==0):
            continue
        i = {'question':question,'docs':docs}
        inputs.append(i)
    '''
    question = query
    
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    docs = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=k,
        score_threshold=score_threshold
    )
    #'''
    #Checking which document is used for the answer
    print("=============================Found Document=============================")
    print(docs) 
    print("=============================Qurey Run=============================")
    # Load and run the QA Chain
    answer = llm_chain.invoke({"docs": docs, "question": question},{"recursion_limit": 1})
    answer = answer.rstrip()
    print("============================OUTPUT================================")
    print(answer)
    print("==============================Qurey End==================================")

    return answer

import asyncio
def runQuery_async(query, persist_directory=persist_directory, model=model_default):
    #print("==============================Prompt=================================")
    prompt = PromptTemplate.from_template(template_query)
    llm = None
    if model in models.keys():
        llm = models[model]
    else:
        print("Model: '"+model+"' not found, using default model.")
        llm = models[model_default]
    #print("==============================CHAIN==================================")
    llm_chain = prompt | llm

    questions = getQuestions(query, llm)
    inputs=[]
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=GPT4AllEmbeddings())
    for question in questions:
        #question = r"Who is the director of Sue & Bill Gross Stem Cell Research Center?"
        docs = vectorstore.similarity_search_with_relevance_scores(
            question,
            k=k,
            score_threshold=score_threshold
        )
        if(len(docs)==0):
            continue
        i = {'question':question,'docs':docs}
        inputs.append(i)
    print("=============================Qurey Run=============================")
    answer = asyncio.run(llm_chain.abatch(inputs,{"recursion_limit": 1}))
    #answer = llm_chain.batch(inputs,{"recursion_limit": 5})
    print("==============================Qurey End==================================")
    print("============================OUTPUT================================")
    print(answer)
    return answer

def getFileList(path):
    if(os.path.isdir(path)):
        files = []
        filepaths = os.listdir(path)
        for filepath in filepaths:
            if(os.path.isdir(path+'\\'+filepath)):
                file = getFileList(path+'\\'+filepath)
                for f in file:
                    files.append(f)
            else:
                files.append(path+'\\'+filepath)
        return files
    else:
        files = []
        files.append(path)
        return files


def main():
    init()
    db = input("Path of database to search: ")
    question = input("Question: ")
    '''
    ts = time.time()
    results1 = runQuery_async(question, db)
    te = time.time()
    t1 = te-ts
    ts = time.time()
    results2 = runQuery_batch(question, db)
    te = time.time()
    t2 = te-ts
    '''
    ts = time.time()
    results3 = runQuery_NoLimit(question, db)
    te = time.time()
    t3 = te-ts
    #print("async_batch:"+str(t1))
    #print("batch:"+str(t2))
    print("Time:"+str(t3))
    '''
    print("============================OUTPUT============================")
    for result in results1:
        print(result1)
    print("=============================END=============================")
    
    print("============================OUTPUT============================")
    for result in results2:
        print(result2)
    print("=============================END=============================")
    '''
    print("============================OUTPUT============================")
    for result in results3:
        print(result)
    print("=============================END=============================")

if __name__ == "__main__":
    if len(sys.argv)==4:
            query = sys.argv[1]
            db = sys.argv[2]
            model = sys.argv[3]
            results = runQuery_NoLimit(query, db, model)
            print("============================OUTPUT============================")
            for result in results:
                print(result)
            print("=============================END=============================")
    elif len(sys.argv)==3:
            query = sys.argv[1]
            db = sys.argv[2]
            results = runQuery_NoLimit(query, db)
            print("============================OUTPUT============================")
            for result in results:
                print(result)
            print("=============================END=============================")
    elif len(sys.argv)==2:
            query = sys.argv[1]
            results = runQuery_NoLimit(query)
            print("============================OUTPUT============================")
            for result in results:
                print(result)
            print("=============================END=============================")
    else:
        main()
else:
    init()
