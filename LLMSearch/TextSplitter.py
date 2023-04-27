import nltk
nltk.download('punkt')
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

def split_text_file(input_file, token_length,init=False):
    with open(input_file, 'r') as f:
        text = f.read()

    tokens = nltk.word_tokenize(text)
    num_tokens = len(tokens)
    num_chunks = num_tokens // token_length + (1 if num_tokens % token_length != 0 else 0)


    def filename(input_file, i):
        output_file = f"{input_file}_chunk_{i}.txt"
        output_file= output_file.replace("/texts/", "/split/")
        return output_file

    output_file =filename(input_file, 0)
    if os.path.exists(output_file) and init==False:
        return

    for i in range(num_chunks):
        start = i * token_length
        end = min((i + 1) * token_length, num_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = ' '.join(chunk_tokens)
        output_file=filename(input_file, i)
        with open(output_file, 'w') as f:
            f.write(chunk_text)

        print(output_file)