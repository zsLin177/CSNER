import json
import random
import os
import argparse

def read_jsonl(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return [json.loads(line) for line in data]

def extract_data(from_data, to_data, tgt_file):
    extracted_data = []
    data_dic = {this_dic["key"]: this_dic for this_dic in to_data}
    for this_dic in from_data:
        key = this_dic['key']
        extracted_data.append(data_dic[key])

    with open(tgt_file, 'w', encoding="utf8") as f:
        for this_dic in extracted_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')

def extract_diff_same_pred(normalize_pred_file, raw_pred_file, same_file, diff_file):
    normalize_pred_data = read_jsonl(normalize_pred_file)
    raw_pred_data = read_jsonl(raw_pred_file)
    same_data = []
    diff_data = []
    for normalize_dic, raw_dic in zip(normalize_pred_data, raw_pred_data):
        assert normalize_dic['key'] == raw_dic['key']
        normal_pred = set([(ne_lst[2], ne_lst[3]) for ne_lst in normalize_dic['label']])
        raw_pred = set([(ne_lst[2], ne_lst[3]) for ne_lst in raw_dic['label']])
        key = normalize_dic['key']
        normal_label_seq_p = normalize_dic['label_seq_p']
        raw_label_seq_p = raw_dic['label_seq_p']
        try:
            normal_txt = normalize_dic['text']
        except:
            normal_txt = normalize_dic['sentences']
        try:
            raw_txt = raw_dic['text']
        except:
            raw_txt = raw_dic['sentences']
        if normal_pred == raw_pred:
            same_data.append({
                'key': key,
                'normal_text': normal_txt,
                'raw_text': raw_txt,
                'label': list(normal_pred),
                'normal_label_seq_p': normal_label_seq_p,
                'raw_label_seq_p': raw_label_seq_p,
            })
        else:
            diff_data.append({
                'key': key,
                'normal_text': normal_txt,
                'raw_text': raw_txt,
                'normal_label': list(normal_pred),
                'raw_label': list(raw_pred),
                'normal_label_seq_p': normal_label_seq_p,
                'raw_label_seq_p': raw_label_seq_p,
            })
    with open(same_file, 'w', encoding="utf8") as f:
        for this_dic in same_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')
    with open(diff_file, 'w', encoding="utf8") as f:
        for this_dic in diff_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')

def generate_goldNE_asrNonNE_and_asrNE_goldNonNE(src_jsonl_file, txt_file, goldNE_asrNonNE_file, asrNE_goldNonNE_file):
    data = read_jsonl(src_jsonl_file)
    goldNE_asrNonNE_data = []
    asrNE_goldNonNE_data = []
    with open(txt_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            goldNE_asrNonNE_txt, asrNE_goldNonNE_txt = line.split('\t')
            goldNE_asrNonNE_data.append(goldNE_asrNonNE_txt)
            asrNE_goldNonNE_data.append(asrNE_goldNonNE_txt)
    assert len(data) == len(goldNE_asrNonNE_data) == len(asrNE_goldNonNE_data)
    with open(goldNE_asrNonNE_file, 'w', encoding='utf8') as f:
        for this_dic, this_txt in zip(data, goldNE_asrNonNE_data):
            this_dic['asr'] = this_txt
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')
    with open(asrNE_goldNonNE_file, 'w', encoding='utf8') as f:
        for this_dic, this_txt in zip(data, asrNE_goldNonNE_data):
            this_dic['asr'] = this_txt
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')

def generate_ramc_train_text_asr(ramc_all_pred_file, ramc_dev_test_file, text_file, asr_file):
    ramc_all_pred_data = read_jsonl(ramc_all_pred_file)
    ramc_dev_test_data = read_jsonl(ramc_dev_test_file)
    ramc_dev_test_dic = {this_dic['key']: this_dic for this_dic in ramc_dev_test_data}
    text = []
    asr = []
    for this_dic in ramc_all_pred_data:
        key = this_dic['key']
        if key in ramc_dev_test_dic:
            continue
        try:
            txt = this_dic['text']
        except:
            txt = this_dic['sentences']
        text.append((key, txt))
        asr.append((key, this_dic['asr']))
    
    with open(text_file, 'w', encoding='utf8') as f:
        for key, txt in text:
            f.write(key + ' ' + txt + '\n')
    with open(asr_file, 'w', encoding='utf8') as f:
        for key, txt in asr:
            f.write(key + ' ' + txt + '\n')

def generate_ramc_train_jsonl(ramc_all_pred_file, ramc_dev_test_file, tgt_file):
    ramc_all_pred_data = read_jsonl(ramc_all_pred_file)
    ramc_dev_test_data = read_jsonl(ramc_dev_test_file)
    ramc_dev_test_dic = {this_dic['key']: this_dic for this_dic in ramc_dev_test_data}
    tgt_data = []
    for this_dic in ramc_all_pred_data:
        key = this_dic['key']
        if key in ramc_dev_test_dic:
            continue
        tgt_data.append(this_dic)
    with open(tgt_file, 'w', encoding='utf8') as f:
        for this_dic in tgt_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')
    
def add_asr(src_file, asr_file, tgt_file):
    src_data = read_jsonl(src_file)
    asr_data = read_jsonl(asr_file)
    asr_dic = {this_dic['key']: this_dic['asr'] for this_dic in asr_data}
    tgt_data = []
    for this_dic in src_data:
        key = this_dic['key']
        this_dic['asr'] = asr_dic[key]
        tgt_data.append(this_dic)
    with open(tgt_file, 'w', encoding='utf8') as f:
        for this_dic in tgt_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')

def generate_distill_label(src_file, tgt_file):
    src_data = read_jsonl(src_file)
    res = []
    for this_dic in src_data:
        key = this_dic['key']
        raw_label = this_dic['label']
        raw_text = this_dic['text']
        asr = this_dic['asr']
        label_seq_p = this_dic['label_seq_p']
        # other_label = this_dic['other_label']
        new_label = map_label_to_text(raw_label, asr)
        res.append({
            'key': key,
            'text': asr,
            'label': new_label,
            'label_seq_p': label_seq_p
        })
    with open(tgt_file, 'w', encoding='utf8') as f:
        for this_dic in res:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')
        
def map_label_to_text(ne_lst, text):
    # use the max cut map to map the label to text
    if len(ne_lst) == 0:
        return []
    res = []
    ne_dict = {}
    for ne in ne_lst:
        ne_dict[ne[3]] = ne[2]
    
    max_ne_len = min(len(text), max([len(key) for key in ne_dict]))
    start = 0
    while start < len(text):
        flag = False
        for i in range(start+max_ne_len, start, -1):
            if text[start:i] in ne_dict:
                res.append([start, i, ne_dict[text[start:i]], text[start:i]])
                start = i
                flag = True
                break
        if not flag:
            start += 1
        max_ne_len = min(len(text)-start, max([len(key) for key in ne_dict]))
    return res

def sample_and_combine(src_file, msra_file, tgt_file, sample_p=0.5, label_p=0):
    random.seed(877)
    src_data = read_jsonl(src_file)
    msra_data = read_jsonl(msra_file)
    res = []
    for this_dic in src_data:
        text = this_dic['text']
        if len(text) == 0:
            continue
        assert " " not in text

        label_seq_p = this_dic['label_seq_p']
        if label_seq_p < label_p:
            continue

        # for instance without ne, sample it
        if len(this_dic['label']) == 0:
            if random.random() < sample_p:
                res.append(this_dic)
        else:
            res.append(this_dic)

    # add all the msra data
    res += msra_data
    with open(tgt_file, 'w', encoding='utf8') as f:
        for this_dic in res:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')

def replace_blank(src_file, tgt_file):
    src_data = read_jsonl(src_file)
    res = []
    num = 0
    for this_dic in src_data:
        asr = this_dic['asr']
        if ' ' in asr:
            num += 1
            asr = asr.replace(' ', '')
        this_dic['asr'] = asr
        res.append(this_dic)
    with open(tgt_file, 'w', encoding='utf8') as f:
        for this_dic in res:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')
    print(num)

def remove_label(src_file, tgt_file):
    src_data = read_jsonl(src_file)
    for this_dic in src_data:
        del this_dic['label']
    with open(tgt_file, 'w', encoding='utf8') as f:
        for this_dic in src_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')

def check_text(src_file):
    data = read_jsonl(src_file)
    for this_dic in data:
        text = this_dic['text']
        if len(text) == 0 or text == ' ' or " " in text:
            print(this_dic['key'])

def split_out_label_in_icsr(src_file, tgt_file):
    src_data = read_jsonl(src_file)
    for this_dic in src_data:
        raw_label = this_dic['label']
        label = []
        other_label = []
        for ne_lst in raw_label:
            if ne_lst[2] in ['ORG', 'PER', 'LOC']:
                label.append(ne_lst)
            else:
                other_label.append(ne_lst)
        this_dic['label'] = label
        this_dic['other_label'] = other_label
    with open(tgt_file, 'w', encoding='utf8') as f:
        for this_dic in src_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')

def add_asr(asr_file, src_file, tgt_file):
    asr_data = read_jsonl(asr_file)
    src_data = read_jsonl(src_file)
    for asr_dic, src_dic in zip(asr_data, src_data):
        assert asr_dic['key'] == src_dic['key']
        try:
            src_dic['asr'] = asr_dic['asr']
        except:
            src_dic['asr'] = asr_dic['text']
    with open(tgt_file, 'w', encoding='utf8') as f:
        for this_dic in src_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')

def analyse(ditill_pred_file, st_pred_file, gold_file, new_distill_pred_file):
    distill_data = read_jsonl(ditill_pred_file)
    st_data = read_jsonl(st_pred_file)
    gold_data = read_jsonl(gold_file)
    in_distill_not_in_st = 0
    in_st_not_in_distill = 0
    new_distill_data = []
    for distill_dic, st_dic, gold_dic in zip(distill_data, st_data, gold_data):
        distill_label = distill_dic['label']
        st_label = st_dic['label']
        gold_label = gold_dic['label']
        txt = distill_dic['text']
        new_distill_label = [] + distill_label
        for ne_lst in gold_label:
            if ne_lst in st_label and ne_lst not in distill_label:
                print(f"asr: {txt}\t{ne_lst} not in distill, but in st")
                new_distill_label.append(ne_lst)
                in_st_not_in_distill += 1
            if ne_lst not in st_label and ne_lst in distill_label:
                print(f"asr: {txt}\t{ne_lst} not in st, but in distill")
                in_distill_not_in_st += 1
        new_distill_label = sorted(new_distill_label, key=lambda x: x[0])
        distill_dic['label'] = new_distill_label
        new_distill_data.append(distill_dic)
    with open(new_distill_pred_file, 'w', encoding='utf8') as f:
        for this_dic in new_distill_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')
    print(f"in_distill_not_in_st: {in_distill_not_in_st}")
    print(f"in_st_not_in_distill: {in_st_not_in_distill}")

def compute_f(pred_file, gold_file):
    pred_data = read_jsonl(pred_file)
    gold_data = read_jsonl(gold_file)
    assert len(pred_data) == len(gold_data)
    tp = 0
    p = 0
    g = 0
    for pred_dic, gold_dic in zip(pred_data, gold_data):
        pred_label = [ne_lst[3] for ne_lst in pred_dic['label']]
        gold_label = [ne_lst[3] for ne_lst in gold_dic['label']]
        p += len(pred_label)
        g += len(gold_label)

        for ne in pred_label:
            if ne in gold_label:
                tp += 1
                gold_label.remove(ne)
    precision = tp / p
    recall = tp / g
    f = 2 * precision * recall / (precision + recall)
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f: {f}")

def count_nes(file_name):
    data = read_jsonl(file_name)
    sum = 0
    for this_dic in data:
        for ne_lst in this_dic['label']:
            if ne_lst[3] != "#" and ne_lst[2] != "NUM":
                sum += 1
    print(sum)

def add_labels(from_file, to_file):
    from_data = read_jsonl(from_file)
    to_data = read_jsonl(to_file)
    res = []
    for from_dic, to_dic in zip(from_data, to_data):
        assert from_dic["key"] == to_dic["key"]
        to_dic["label"] = from_dic["label"]
        to_dic["other_label"] = from_dic["other_label"]
        res.append(to_dic)
    with open(to_file, "w", encoding="utf8") as f:
        for dic in to_data:
            f.write(json.dumps(dic, ensure_ascii=False) + "\n")

def random_sample(src_file, tgt_file, sample_num=500):
    src_data = read_jsonl(src_file)
    tgt_data = random.sample(src_data, sample_num)
    with open(tgt_file, "w", encoding="utf8") as f:
        for dic in tgt_data:
            f.write(json.dumps(dic, ensure_ascii=False) + "\n")

def aishell_trans(src_jsonl_file, tgt_jsonl_file):
    src_data = read_jsonl(src_jsonl_file)
    tgt_data = []
    for this_dic in src_data:
        key = this_dic['key']
        text = this_dic['txt']
        wav = this_dic['wav']
        raw_label = this_dic['ne_lst']
        label = []
        for ne in raw_label:
            label.append([ne[1], ne[2]+1, ne[0], text[ne[1]:ne[2]+1]])

        tgt_data.append({
            'key': key,
            'text': text,
            'label': label,
            'wav': wav,
        })
    with open(tgt_jsonl_file, 'w', encoding='utf8') as f:
        for this_dic in tgt_data:
            f.write(json.dumps(this_dic, ensure_ascii=False) + '\n')

def mapping_and_combine(tea_pred_file, asr_file, src_file, tgt_file):
    tmp_file_1 = os.path.join(os.path.dirname(tea_pred_file), "tmp1.jsonl")
    add_asr(asr_file, tea_pred_file, tmp_file_1)
    tmp_file_2 = os.path.join(os.path.dirname(tea_pred_file), "tmp2.jsonl")
    generate_distill_label(tmp_file_1, tmp_file_2)
    sample_and_combine(tmp_file_2, src_file, tgt_file, sample_p=0.5)
    os.remove(tmp_file_1)
    os.remove(tmp_file_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, required=True)
    parser.add_argument("--tgt_file", type=str, required=True)
    parser.add_argument("--tea_pred_file", type=str, required=True)
    parser.add_argument("--asr_file", type=str, required=True)
    args = parser.parse_args()
    mapping_and_combine(args.tea_pred_file, args.asr_file, args.src_file, args.tgt_file)
    
    