# trun the  same json file to conll format

import json
import os

def json2bio(json_file, bio_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        res = []
        for line in lines:
            this_dic = json.loads(line)
            sentence = this_dic['aishell']['sentence']
            entity = this_dic['aishell']['entity']
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

if __name__ == '__main__':
    json2bio('same_data.json', 'tmp.bies')
    bio2conll('tmp.bies', 'same_data.conll')
    os.remove('tmp.bies')
