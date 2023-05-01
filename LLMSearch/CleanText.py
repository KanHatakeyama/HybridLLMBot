"""
original code partially from
https://qiita.com/Cartelet/items/6dcf525db8c3a7953524

"""
import re
import unicodedata


def pad_text(text,chunk_size_limit=100):
    t=""
    while True:
        t+=text+"."
        if len(t)>chunk_size_limit:
            break
    return t

def len_(text):
    cnt = 0
    for t in text:
        if unicodedata.east_asian_width(t) in "FWA":
            cnt += 2
        else:
            cnt += 1
    return cnt

def condense_characters(text):
    text = re.sub(r' +', ' ', text)
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\.+', '.', text)
    return text

def remove_two_byte_spaces(text):
    # 全角文字だけの単語間のスペースを削除
    text = re.sub(r'(?<=[\u3000-\u9FFF])[\u3000\s]+(?=[\u3000-\u9FFF])', '', text)
    text=condense_characters(text)
    return text


def clean_text(text, n=30, bracketDetect=True):
    text = text.splitlines()
    sentences = []
    t = ""
    bra_cnt = ket_cnt = bra_cnt_jp = ket_cnt_jp = 0
    for i in range(len(text)):
        if not bool(re.search("\S", text[i])): continue
        if bracketDetect:
            bra_cnt += len(re.findall("[\(（]", text[i]))
            ket_cnt += len(re.findall("[\)）]", text[i]))
            bra_cnt_jp += len(re.findall("[｢「『]", text[i]))
            ket_cnt_jp += len(re.findall("[｣」』]", text[i]))
        if i != len(text) - 1:
            if bool(re.fullmatch(r"[A-Z\s]+", text[i])):
                if t != "": sentences.append(t)
                t = ""
                sentences.append(text[i])
            elif bool(
                    re.match(
                        "(\d{1,2}[\.,、．]\s?(\d{1,2}[\.,、．]*)*\s?|I{1,3}V{0,1}X{0,1}[\.,、．]|V{0,1}X{0,1}I{1,3}[\.,、．]|[・•●])+\s",
                        text[i])) or re.match("\d{1,2}．\w", text[i]) or (
                            bool(re.match("[A-Z]", text[i][0]))
                            and abs(len_(text[i]) - len_(text[i + 1])) > n
                            and len_(text[i]) < n):
                if t != "": sentences.append(t)
                t = ""
                sentences.append(text[i])
            elif (
                    text[i][-1] not in ("。", ".", "．") and
                (abs(len_(text[i]) - len_(text[i + 1])) < n or
                 (len_(t + text[i]) > len_(text[i + 1]) and bool(
                     re.search("[。\.．]\s\d|..[。\.．]|.[。\.．]", text[i + 1][-3:])
                     or bool(re.match("[A-Z]", text[i + 1][:1]))))
                 or bool(re.match("\s?[a-z,\)]", text[i + 1]))
                 or bra_cnt > ket_cnt or bra_cnt_jp > ket_cnt_jp)):
                t += text[i]
            else:
                sentences.append(t + text[i])
                t = ""
        else:
            sentences.append(t + text[i])

    final_text="\n".join(sentences)
    final_text=remove_two_byte_spaces(final_text)
    return final_text 
