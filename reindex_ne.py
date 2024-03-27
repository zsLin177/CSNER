import json
import re
import os
import string
from zhon.hanzi import punctuation

def read_jsonl(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        lines = f.readlines()
        res = []
        for line in lines:
            res.append(json.loads(line))
    return res

def get_matched_index(sentence, entities):
    # sentence: str
    # entities: list of dict
    # return: list of list
    new_entities = []
    for entity in entities:
        label = entity[0]
        word = entity[3]
        matches = [(m.start(), m.end()-1) for m in re.finditer(word, sentence)]
        if len(matches) > 0:
            for match in matches:
                new_entities.append([label, match[0], match[1], word])
    return new_entities

def add_re_index_entity(file_name, contain_punc=False):
    raw_res = read_jsonl(file_name)
    punctuations = string.punctuation + punctuation
    for item in raw_res:
        if not contain_punc:
            new_s = ''
            for char in item['speak_style']:
                if char not in punctuations:
                    new_s += char
            item['speak_style'] = new_s
        item['speak_style_entity'] = get_matched_index(item['speak_style'], item['entity'])
    return raw_res
    
def write_jsonl(file_name, data):
    with open(file_name, 'w', encoding='utf8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def json2bio(json_file, bio_file, normal):
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        res = []
        for line in lines:
            this_dic = json.loads(line)
            if normal:
                sentence = this_dic['sentence']
                entity = this_dic['entity']
            else:
                sentence = this_dic['speak_style']
                entity = this_dic['speak_style_entity']
            bio_lst = ['O'] * len(sentence)
            for ne in entity:
                bio_lst[ne[1]] = 'B-' + ne[0]
                for i in range(ne[1]+1, ne[2]+1):
                    bio_lst[i] = 'I-' + ne[0]
            for i, token in enumerate(sentence):
                res.append(token + ' ' + bio_lst[i] + '\n')
            res.append('\n')
    with open(bio_file, 'w', encoding='utf-8') as f:
        f.writelines(res)

def bio2conll(bio_file, conll_file):
    with open(bio_file, 'r', encoding='utf-8') as f:
        line_lsts = [line.strip() for line in f]

    sentence_lsts = []
    start, i = 0, 0
    for line in line_lsts:
        if len(line) < 1:
            sentence_lsts.append(line_lsts[start:i])
            start = i + 1
        i += 1

    with open(conll_file, 'w', encoding='utf-8') as f:
        for sent_lst in sentence_lsts:
            for i, line_s in enumerate(sent_lst, 1):
                token, alllabel = line_s.split()
                if alllabel == 'O':
                    real_label = 'O'
                elif alllabel.split('-')[1] == 'NR':
                    real_label = alllabel.split('-')[0] + '-PER'
                elif alllabel.split('-')[1] == 'NT':
                    real_label = alllabel.split('-')[0] + '-ORG'
                elif alllabel.split('-')[1] == 'NS':
                    real_label = alllabel.split('-')[0] + '-LOC'
                else:
                    real_label = alllabel
                new_line = str(i) + '\t' + token + '\t_\t' + real_label + '\t' + real_label + '\t_\t_\t_\t_\t_\n'
                f.write(new_line)
            f.write('\n')


def toconll(jsonl_file, conll_file, normal):
    json2bio(jsonl_file, 'tmp.bies', normal)
    bio2conll('tmp.bies', conll_file)
    os.remove('tmp.bies')

if __name__ == "__main__":
    data = add_re_index_entity('same_ext254_with_speak.jsonl')
    write_jsonl('same_ext254_with_speak_nes.jsonl', data)

    toconll('same_ext254_with_speak_nes.jsonl', 'same_ext254_with_speak_nes.conll', False)
    toconll('same_ext254_with_speak_nes.jsonl', 'same_ext254.conll', True)