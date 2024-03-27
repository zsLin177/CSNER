import json
import os
import argparse

def read_jsonl(file_name):
    res = []
    with open(file_name, 'r', encoding="utf8") as f:
        for s in f.readlines():
            res.append(json.loads(s))
    return res

def look_data(jsonl_file_name, conll_file_name):
    data = read_jsonl(jsonl_file_name)
    utterance_num = len(data)

    sum_ne_num = get_ne_from_conll(conll_file_name)
    ratio = sum_ne_num / utterance_num
    max_uttr_len = max([len(item["sentences"]) for item in data])
    min_uttr_len = min([len(item["sentences"]) for item in data])
    if ratio < 1 or max_uttr_len > 1000:
        return
    print("file_name: ", jsonl_file_name)
    print("utterance_num: ", utterance_num)
    print("max_uttrance_len: ", max_uttr_len)
    print("min_uttrance_len: ", min_uttr_len)
    sum_ne_num = get_ne_from_conll(conll_file_name)
    print("avg ne_num per utterance ", sum_ne_num / utterance_num)
    print()

def get_ne_from_conll(conll_file_name):
    with open(conll_file_name, 'r', encoding="utf8") as f:
        lines = f.readlines()
        sentence_lst = []
        start = 0
        for i, line in enumerate(lines):
            if line == "\n":
                sentence_lst.append(lines[start:i])
                start = i + 1

    sum_ne_num = 0
    for sent_lst in sentence_lst:
        label_lst = []
        for line in sent_lst:
            line = line.strip()
            line_lst = line.split("\t")
            label_lst.append(line_lst[4])

        ne_lst = []
        start = 0
        flag = False
        ne_label = ""
        for i, label in enumerate(label_lst):
            if label.startswith("B-"):
                if flag:
                    ne_lst.append((start, i, ne_label))
                    flag = False
                flag = True
                ne_label = label[2:]
                start = i
            elif label.startswith("I-"):
                continue
            else:
                if flag:
                    ne_lst.append((start, i, ne_label))
                    flag = False
        if flag:
            ne_lst.append((start, i+1, ne_label))
        sum_ne_num += len(ne_lst)
    return sum_ne_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", type=str)
    parser.add_argument("--conll_dir", type=str)
    args = parser.parse_args()
    for file_name in os.listdir(args.jsonl_dir):
        if not file_name.startswith("CTS"):
            continue
        jsonl_file_name = os.path.join(args.jsonl_dir, file_name)
        conll_file_name = os.path.join(args.conll_dir, file_name.split(".")[0] + ".conll.pred")
        look_data(jsonl_file_name, conll_file_name)