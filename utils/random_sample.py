# random sample examples from the data
import json
import random

def read_json(file_name):
    res = []
    with open(file_name, 'r', encoding="utf8") as f:
        for s in f.readlines():
            res.append(json.loads(s))
    return res

def random_sample(file_name, sample_size, output_file1, output_file2):
    data = read_json(file_name)
    random.shuffle(data)
    data1 = data[:sample_size]
    with open(output_file1, 'w', encoding="utf8") as f:
        for item in data1:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    data2 = data[sample_size:]
    with open(output_file2, 'w', encoding="utf8") as f:
        for item in data2:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

        