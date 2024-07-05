import sys
import time
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain

#from insertrecord_test import *
from insertrecord import *

# Default model path
models_path = r".\models"
model_default = r"Llama-3-11.5B-Instruct-Coder-v2-Q8_0"
model_split = r"Llama-3-11.5B-Instruct-Coder-v2-Q8_0" #當時強制使用此模型因為其他模型用來拆解大問題的效果不好
#Llama-3-11.5B-Instruct-Coder-v2-Q8_0
#Meta-Llama-3-8B-Instruct.Q4_0
#Yi-1.5-34B-Chat-Q8_0
# Default database path
persist_directory=r".\db"

# Prompt templates
template_query = ("System: Given the documents {docs}\n\n" +
    "User: {question}\n\n" +
    "Assistant: According to the documents, "
    )
'''
template_split = (
    "User: If there are multiple sentences or questions in the following, \n\n{question}\n\n" + 
    "Split them into individual lines and surround each sentences with semicolon in both beginning and end. " +
    "\n\nAssistant:"
    )
'''
template_split = (
    "User: if there multiple commands or questions in the sentence \"{question}\"." + 
    "Rephrase it into multiple sentances in individual lines and surround them with semicolon in both beginning and end. " +
    "\n\nAssistant:"
    )
# Parameter for llm
#n_gpu_layers = -1  # Metal set to 1 is enough. unused
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
max_tokens = 300
temperature = 0
top_p = 0.1

# parameter for similarity test
k = 1 # number of document will be used
score_threshold = 0.6 #1 is higher the similarity, 0 is lower.

# String to return when there is no answer for the question
ans_null = "I'm sorry, but I currently don't have the information you're seeking."
#use "{question}" as the question provided

N = 3

models = {}

def init():
    path_list = getFileList(models_path)
    for path in path_list:
        name = Path(path).stem
        #WARNING ISSUE: https://github.com/langflow-ai/langflow/issues/1820
        llm = LlamaCpp(
            #n_gpu_layers = n_gpu_layers,
            model_path=path,
            n_batch=n_batch,
            n_ctx=2048,
            temperature = temperature, 
            max_tokens = max_tokens,
            top_p=top_p,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            verbose=True, # May need to set to True, somehow it breaks when False...
        )
        models[name]=llm
        print("Model: '"+name+"' loaded")

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

def runQuery(query, persist_directory=persist_directory, model=model_default):
    """
    LOG
    """
    data = {
        'question': query,
        'answer': 'answer',
        'date_enter': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # Compute the time elapsed since start time and additional 2 seconds
        'time_elape': round((time.time() - datetime.now().timestamp()) + 2), # Adjusted to match the context
        'temperature': temperature,
        'topp': top_p,
        'topk': k,
        'model_name': model,
        #source
        'local_rag': persist_directory,
        'con_score': 0,
        #text logs
        'misc': []
    }
    score = []
    misc = []
    time_start = time.time()
    """
    LOG
    """
    results = []
    #print("==============================Prompt=================================")
    prompt = PromptTemplate.from_template(template_query)
    llm = None
    if model in models.keys():
        llm = models[model]
    else:
        llm = models[model_default]
        print("Model: '"+model+"' not found, using default model.")
        """
        LOG
        """
        misc.append("Model: '"+model+"' not found, using default model.");
        """
        LOG
        """
    #print("==============================CHAIN==================================")
    llm_chain = prompt | llm.bind(stop=["User:"])

    questions = getQuestions(query, llm)
    """
    LOG
    """
    misc.append("Splitted questions: "+ str(questions));
    """
    LOG
    """
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=GPT4AllEmbeddings())
    
    for question in questions:
        #question = r"Who is the director of Sue & Bill Gross Stem Cell Research Center?"
        docs = vectorstore.similarity_search_with_relevance_scores(
            question,
            k=k,
            score_threshold=score_threshold
        )
        print("Query: "+str(question))
        #Checking which document is used for the answer
        print("=============================Found Document=============================")
        print(docs)
        if(len(docs)==0):
            #results.append(ans_null.replace("{question}",question))
            continue
        """
        LOG
        """
        #imp = True
        for doc in docs:
            #if imp: #put the first doc name only
            #    #print(doc[0].metadata['source'])
            #    data['local_rag']=str(doc[0].metadata['source']).strip()
            #    imp = False
            misc.append(doc[0].page_content)
            score.append(doc[1])
        """
        LOG
        """
        
        print("=============================Qurey Run=============================")
        inputs = []
        #for doc in docs:
        #    inputs.append(doc[0])
        # Load and run the QA Chain
        answer = llm_chain.invoke({"docs": docs, "question": question},{"recursion_limit": 1})
        answer = answer.rstrip()
        print("============================OUTPUT================================")
        print(answer)
        print("==============================Qurey End==================================")
        results.append(answer)
    if(len(results)==0):
        results.append(ans_null.replace("{question}",query))
    #print(results)
    print("==============================Finished==================================")
    """
    LOG
    """
    #te = time.time()
    count = 0
    total_score = 0.0
    for s in score:
        total_score+=s
        count+=1
        
    #str(local_rag)
    data['answer'] = str(results)
    data['model_name']=model
    if count!=0:
        data['con_score']=round((total_score/count),3)
    data['misc']=str(misc)
    data['time_elape']=round((time.time() - time_start) + 2)
    data['date_enter']=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #insertRec(query,str(results),temperature,top_p,k,model,"",(total_score/count),str(misc),(te-t))
    
    try:
        insert_record(data)
    except Exception as e:
        print(e)
    
    """
    LOG
    """
    return results
    
def runQuery_batch(query, persist_directory=persist_directory, model=model_default):
    results = []
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
    answer = llm_chain.batch(inputs,{"recursion_limit": 1})
    answer = answer.rstrip()
    print("==============================Qurey End==================================")
    print("============================OUTPUT================================")
    print(answer)
    return answer

def runSingleQuery(query, persist_directory=persist_directory, model=model_default):
    """
    LOG
    """
    data = {
        'question': query,
        'answer': 'answer',
        'date_enter': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # Compute the time elapsed since start time and additional 2 seconds
        'time_elape': round((time.time() - datetime.now().timestamp()) + 2), # Adjusted to match the context
        'temperature': temperature,
        'topp': top_p,
        'topk': k,
        'model_name': model,
        #source
        'local_rag': persist_directory,
        'con_score': 0,
        #text logs
        'misc': []
    }
    score = []
    misc = []
    time_start = time.time()
    """
    LOG
    """

    results = []
    #print("==============================Prompt=================================")
    prompt = PromptTemplate.from_template(template_query)
    llm = None
    if model in models.keys():
        llm = models[model]
    else:
        print("Model: '"+model+"' not found, using default model.")
        llm = models[model_default]
        """
        LOG
        """
        misc.append("Model: '"+model+"' not found, using default model.");
        """
        LOG
        """
    #print("==============================CHAIN==================================")
    llm_chain = prompt | llm.bind(stop=["User:"])

    question = query
    
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=GPT4AllEmbeddings())

    #question = r"Who is the director of Sue & Bill Gross Stem Cell Research Center?"
    docs = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=k,
        score_threshold=score_threshold
    )
    #'''
    #Checking which document is used for the answer
    print("=============================Found Document=============================")
    print(docs) 
    if(len(docs)==0):
        answer = ans_null.replace("{question}",question)
        print(answer)
        return answer
    """
    LOG
    """
    imp = True
    for doc in docs:
        #if imp: #put the first doc name only
        #    #print(doc[0].metadata['source'])
        #    data['local_rag']=str(doc[0].metadata['source']).strip()
        #    imp = False
        misc.append(doc[0].page_content)
        score.append(doc[1])
    """
    LOG
    """
    print("=============================Qurey Run=============================")
    # Load and run the QA Chain
    answer = llm_chain.invoke({"docs": docs, "question": question},{"recursion_limit": 1})
    answer = answer.rstrip()
    print("============================OUTPUT================================")
    print(answer)
    print("==============================Qurey End==================================")
    """
    LOG
    """
    #te = time.time()
    count = 0
    total_score = 0.0
    for s in score:
        total_score+=s
        count+=1
        
    #str(local_rag)
    data['answer'] = str(answer)
    data['model_name']=model
    if count!=0:
        data['con_score']=round((total_score/count),3)
    data['misc']=str(misc)
    data['time_elape']=round((time.time() - time_start) + 2)
    data['date_enter']=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #insertRec(query,str(results),temperature,top_p,k,model,"",(total_score/count),str(misc),(te-t))
    
    try:
        insert_record(data)
    except Exception as e:
        print(e)
    
    """
    LOG
    """
    return answer

import asyncio
def runQuery_async(query, persist_directory=persist_directory, model=model_default):
    results = []
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
    results3 = runQuery(question, db)
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
    #print("============================OUTPUT============================")
    #for result in results3:
    #    print(result)
    #print("=============================END=============================")

if __name__ == "__main__":
    if len(sys.argv)==4:
            query = sys.argv[1]
            persist_directory = sys.argv[2]
            model = sys.argv[3]
            results = runQuery(query, db, model)
            print("============================OUTPUT============================")
            for result in results:
                print(result)
            print("=============================END=============================")
    elif len(sys.argv)==3:
            query = sys.argv[1]
            persist_directory = sys.argv[2]
            results = runQuery(query, db)
            print("============================OUTPUT============================")
            for result in results:
                print(result)
            print("=============================END=============================")
    elif len(sys.argv)==2:
            query = sys.argv[1]
            results = runQuery(query)
            print("============================OUTPUT============================")
            for result in results:
                print(result)
            print("=============================END=============================")
    else:
        main()
else:
    init()