#persist_directory = r"./db2" # path to save the vector data
chunk_size = 1000 # number of characters will try to split (may be exceed when paragraph is not ended)
chunk_overlap = 400 # for orignal text split method

import sys
import os
import shutil
#from langchain.docstore.document import Document
#from TXTLoader import loadFile as TXTLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

local_model_path= "./embedding_model/bge-m3"

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

def load(PATH, persist_directory):
    #delete old database, remove this if not needed
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"The directory {persist_directory} has been deleted.")
    
    # Load docs
    fileList = getFileList(PATH)
    
    #'''
    # Original method from langchain
    doclist = []
    for file in fileList:
        print("Loading file - " + file)
        loader = None
        if(file.endswith(".txt")):
            loader = TextLoader(file)
            data = loader.load()
            for d in data:
                doclist.append(d)
        if(file.endswith(".pdf")):
            loader = PyPDFLoader(file)
            data = loader.load()
            for d in data:
                doclist.append(d)
 
    print("載入文件完成, 將開始讀取文件")
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    #text_splitter = CharacterTextSplitter(separator="\n",chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(doclist)
    '''
    
    all_splits = []
    for file in fileList:
        print("Loading file - " + file)
        if(file.endswith(r".txt")):
            data = TXTLoader(file, chunk_size, chunk_overlap)
            for d in data:
                print(d)
                all_splits.append(d)
                #print(d)
        #TODO: add support for other file type
        
    '''    
    # Store splits
    

    #store the vector data with cosine score of later similarity test
    vectorstore = Chroma.from_documents(
        documents=all_splits, 
        embedding = HuggingFaceEmbeddings(model_name=local_model_path,model_kwargs = {'device': 'cuda:1'}),#GPT4AllEmbeddings(),#HuggingFaceEmbeddings(model_name=local_model_path,model_kwargs = {'device': 'cuda:1'})
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Finished.")

if __name__ == "__main__":
    path = ''
    if len(sys.argv)>2:
        path = sys.argv[1]
        persist_directory = sys.argv[2]
    else:
        print("Alternative usage: python load_files.py <Path of data> <Path of database>")
        path = input("Please input path of file or folder: ")
        persist_directory = input("Please input path of database: ")
    print("Loading from path - "+ path)
    load(path, persist_directory)
