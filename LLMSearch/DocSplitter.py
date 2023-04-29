import json
import os
from llama_index import SimpleDirectoryReader
import glob
import re

def filename_fn(filename): return {"file_name": filename}

def print_v(text,verbose):
    if verbose:
        print(text)

def split_documents(json_path='settings/settings.json',initiate=False,verbose=True):
    with open(json_path) as f:
        settings = json.load(f)


    original_folder_path=settings["data_path"]+"/original"
    split_folder_path=settings["data_path"]+"/split"
    chunk_size_limit=settings["chunk_size_limit"]


    directory_reader = SimpleDirectoryReader(
        original_folder_path,
        file_metadata=filename_fn,
        recursive=True,
    )

    available_file_list= [str(input_file)
                    for input_file in directory_reader.input_files]   

    if initiate:
        #delete all files in split folder
        split_file_list=glob.glob(split_folder_path+"/*")
        for file in split_file_list:
            os.remove(file)


    #check for finished documents
    split_file_list=glob.glob(split_folder_path+"/*")
    fin_file_list=[re.sub(r"_[0-9]+.txt","",file) for file in split_file_list]
    fin_file_list=list(set(fin_file_list))
    fin_file_list=[os.path.basename(file) for file in fin_file_list]
    fin_file_list=[original_folder_path+"/"+file for file in fin_file_list]

    fin_ids=[]
    for i,file in enumerate(available_file_list):
        if file in fin_file_list:
            fin_ids.append(i)
            print_v("skip: "+file,verbose)

    for i in sorted(fin_ids, reverse=True):
            directory_reader.input_files.pop(i)


    documents = directory_reader.load_data()

    for doc in documents:
        text=doc.text
        file_path=doc.extra_info["file_name"]
        file_name=os.path.basename(file_path)

        chunk_list=split_text(text,chunk_size_limit)

        for i,chunk in enumerate(chunk_list):
            output_file=split_folder_path+"/"+file_name+"_"+str(i)+".txt"
            with open(output_file, 'w') as f:
                f.write(chunk)
        
        print_v("split: "+file_name,verbose)



def unify_text(text):
    text=text.replace("。","\n")
    text=text.replace(".","\n")
    text=text.replace("．","\n")
    text=text.replace("？","\n")
    text=text.replace("?","\n")
    text=text.replace("！","\n")
    text=text.replace("!","\n")
    text=text.replace("；","\n")
    text=text.replace(";","\n")
    text=text.replace("：","\n")
    text=text.replace(":","\n")
    text=text.replace("　"," ")

    return text


def split_text(text,chunk_size_limit=100):
    text=unify_text(text)

    text_list=text.split("\n")

    chunk_text_list=[]
    temp_text=""
    for t in text_list:
        t=t.strip()
        temp_text+=t+"."
        if len(temp_text)<chunk_size_limit:
            continue
        else:
            chunk_text_list.append(temp_text)
            temp_text=""

    return chunk_text_list

def pad_text(text,chunk_size_limit=100):
    t=""
    while True:
        t+=text+"."
        if len(t)>chunk_size_limit:
            break
    return t