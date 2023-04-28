#import nltk
#nltk.download('punkt')
import glob
import os
import shutil

def auto_txt_split(glob_path, chunk_size_limit,init=False):
    path_list=glob.glob(glob_path)

    if init:
        #delete split files
        split_path=glob_path.replace("/texts/", "/split/")
        folder_path=os.path.dirname(split_path)
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)

    for path in path_list:
        split_text_file(path, chunk_size_limit,init=init)

def split_text_file(input_file, chunk_size_limit=100,init=False):
    with open(input_file, 'r') as f:
        text = f.read()


    def filename(input_file, i):
        output_file = f"{input_file}_chunk_{i}.txt"
        output_file= output_file.replace("/texts/", "/split/")
        return output_file

    output_file =filename(input_file, 0)
    if os.path.exists(output_file) and init==False:
        return

    text_list=split_text(text,chunk_size_limit)
    for i,chunk_text in enumerate(text_list):
       output_file=filename(input_file, i)
       with open(output_file, 'w') as f:
            f.write(chunk_text)

       print(output_file)



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

