import json
import os
from llama_index import SimpleDirectoryReader
import glob
import re
from .CleanText import clean_text
import hashlib
from tqdm import tqdm


def filename_fn(filename): return {"file_name": filename}


def print_v(text, verbose):
    if verbose:
        print(text)


def split_documents(json_path='settings/settings.json', initiate=False, verbose=True):
    with open(json_path) as f:
        settings = json.load(f)

    original_folder_path = settings["data_path"]+"/original"
    split_folder_path = settings["data_path"]+"/split"
    fin_text_path = settings["data_path"]+"/meta/text_finished.txt"
    chunk_size_limit = settings["chunk_size_limit"]

    if not os.path.exists(split_folder_path):
        os.mkdir(split_folder_path)

    directory_reader = SimpleDirectoryReader(
        original_folder_path,
        file_metadata=filename_fn,
        recursive=True,
    )

    if initiate:
        # delete all files in split folder
        split_file_list = glob.glob(split_folder_path+"/*")
        print_v("removing files...", verbose)
        for file in tqdm(split_file_list):
            os.remove(file)

    # load unfinished files
    with open(fin_text_path, 'r') as f:
        fin_file_list = f.read().splitlines()

    print_v("checking for finished files...", verbose)
    fin_ids = []
    for i, file in enumerate(directory_reader.input_files):
        # print(file)
        if str(file) in fin_file_list:
            fin_ids.append(i)

    for i in sorted(fin_ids, reverse=True):
        directory_reader.input_files.pop(i)

    documents = directory_reader.load_data()

    # log finished files
    mode = "a" if not initiate else "w"
    with open(fin_text_path, mode) as f:
        for path in directory_reader.input_files:
            f.write(str(path)+"\n")

    print_v("splitting...", verbose)
    for doc in tqdm(documents):
        text = doc.text
        text = clean_text(text)
        file_path = doc.extra_info["file_name"]
        file_name = os.path.basename(file_path)

        chunk_list = split_text(text, chunk_size_limit)
        # print(file_path)
        for i, chunk in enumerate(chunk_list):
            #print(i, chunk)
            base_name = split_folder_path+"/"+file_name+"_"+str(i)
            output_file = base_name+"_" + \
                generate_unique_code(base_name+file_path)+".txt"
            with open(output_file, 'w') as f:
                f.write(chunk)

        #print_v("split: "+file_name, verbose)


def unify_text(text):
    text = text.replace("。", "\n")
    text = text.replace(".", "\n")
    text = text.replace("．", "\n")
    text = text.replace("？", "\n")
    text = text.replace("?", "\n")
    text = text.replace("！", "\n")
    text = text.replace("!", "\n")
    text = text.replace("；", "\n")
    text = text.replace(";", "\n")
    text = text.replace("：", "\n")
    text = text.replace(":", "\n")
    text = text.replace("　", " ")

    return text


def split_text(text, chunk_size_limit=100):
    text = unify_text(text)

    text_list = text.split("\n")

    chunk_text_list = []
    temp_text = ""
    for t in text_list:
        t = t.strip()
        temp_text += t+"."
        if len(temp_text) < chunk_size_limit:
            continue
        else:
            chunk_text_list.append(temp_text)
            temp_text = ""

    chunk_text_list.append(temp_text)
    return chunk_text_list


def pad_text(text, chunk_size_limit=100):
    t = ""
    while True:
        t += text+"."
        if len(t) > chunk_size_limit:
            break
    return t


def generate_unique_code(text, length=10):
    # 文字列をバイト列に変換
    text_bytes = text.encode('utf-8')

    # SHA-256ハッシュを計算
    hash_object = hashlib.sha256(text_bytes)

    # ハッシュを16進数の文字列に変換
    unique_code = hash_object.hexdigest()

    return unique_code[:length]
