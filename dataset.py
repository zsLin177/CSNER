import json
import torch
from transformers import AutoTokenizer
from supar.utils.fn import pad

def read_json(file_name, filter_keys=set()):
    res = []
    with open(file_name, 'r') as f:
        for s in f.readlines():
            this_dict = json.loads(s)
            if this_dict["key"] in filter_keys:
                continue
            res.append(this_dict)
    return res

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            file_name,
            tokenizer_path,
            label_lst=['O', 'B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER'],
            astrain=False,
            filter_keys=set(),
            onasr=False,
            icsr=False,
    ):
        self.data = read_json(file_name, filter_keys=filter_keys)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.label_itos = label_lst
        self.label_stoi = {label: i for i, label in enumerate(label_lst)}
        self.astrain = astrain
        self.onasr = onasr
        self.icsr = icsr
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        key = self.data[index]["key"]
        if not self.onasr:
            try:
                sentence_to_parse = self.data[index]["sentences"]
            except:
                sentence_to_parse = self.data[index]["text"]
        else:
            try:
                sentence_to_parse = self.data[index]["asr"]
            except:
                raise Exception("onasr is True but no asr in data")
        try:
            asr = self.data[index]["asr"]
        except:
            asr = None

        try:
            gold_s = self.data[index]["sentences"]
        except:
            gold_s = self.data[index]["text"]
        
        tokenids = [self.tokenizer.cls_token_id]
        # tokenids: len(sentence) + 1 (cls, xx, ...)
        for word in sentence_to_parse:
            word_tokenids = self.tokenizer.encode(word, add_special_tokens=False)
            tokenids += word_tokenids

        if not self.astrain:
            labelids = None
        else:
            assert "label" in self.data[index]
            # labelids: len(sentence)
            labelids = [self.label_stoi["O"]] * len(gold_s)
            for ne in self.data[index]["label"]:
                st, ed = ne[0], ne[1] # do not contain ed
                ne_label = ne[2]
                for i in range(st, ed):
                    if i == st:
                        labelids[i] = self.label_stoi["B-" + ne_label]
                    else:
                        labelids[i] = self.label_stoi["I-" + ne_label]
        
        label_pad_id = self.label_stoi["O"]

        if self.icsr:
            ins_label = "其他"
            try:
                assert "other_label" in self.data[index]
                for o_label in self.data[index]["other_label"]:
                    if o_label[3] == "#":
                        ins_label = o_label[2]
                        break
            except:
                ins_label = None
        else:
            ins_label = None

        return key, tokenids, labelids, sentence_to_parse, label_pad_id, asr, gold_s, ins_label
    
def collate_fn(batch):
    keys = [item[0] for item in batch]
    parse_sents = [item[3] for item in batch]
    asr = [item[5] for item in batch]
    gold_s = [item[6] for item in batch]
    if batch[0][7] is not None:
        ins_labels = [item[7] for item in batch]
    else:
        ins_labels = None
    label_pad_id = batch[0][4]
    tokenid_tensors = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    tokenid_tensors = pad(tokenid_tensors, padding_value=0)
    # [batch_size, max_len, 1]
    tokenid_tensors = tokenid_tensors.unsqueeze(-1)
    if batch[0][2] is not None:
        labelid_tensors = [torch.tensor(item[2], dtype=torch.long) for item in batch]
        # [batch_size, max_len]
        labelid_tensors = pad(labelid_tensors, padding_value=label_pad_id)
    else:
        labelid_tensors = None
    
    return {"keys": keys, "tokenids": tokenid_tensors, "labelids": labelid_tensors, "parse_sents": parse_sents, "asr": asr, "gold_s": gold_s, "ins_labels": ins_labels}

class InsDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            file_name,
            tokenizer_path,
            label_lst=["导航", "空调", "联系", "音乐", "其他"],
            astrain=False,
            onasr=False,
    ):
        self.data = read_json(file_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.label_itos = label_lst
        self.label_stoi = {label: i for i, label in enumerate(label_lst)}
        self.astrain = astrain
        self.onasr = onasr

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        key = self.data[index]["key"]
        if not self.onasr:
            sentence_to_parse = self.data[index]["text"]
        else:
            try:
                sentence_to_parse = self.data[index]["asr"]
            except:
                raise Exception("onasr is True but no asr in data")
        try:
            asr = self.data[index]["asr"]
        except:
            asr = None
        gold_s = self.data[index]["text"]

        tokenids = [self.tokenizer.cls_token_id]
        # tokenids: len(sentence) + 1 (cls, xx, ...)
        for word in sentence_to_parse:
            word_tokenids = self.tokenizer.encode(word, add_special_tokens=False)
            tokenids += word_tokenids
        
        if not self.astrain:
            labelid = None
        else:
            assert "other_label" in self.data[index]
            for o_label in self.data[index]["other_label"]:
                if o_label[3] == "#":
                    assert o_label[2] in self.label_itos
                    labelid = self.label_stoi[o_label[2]]
                    break
        return key, tokenids, labelid, sentence_to_parse, asr, gold_s

def ins_collate_fn(batch):
    keys = [item[0] for item in batch]
    parse_sents = [item[3] for item in batch]
    asr = [item[4] for item in batch]
    gold_s = [item[5] for item in batch]
    tokenid_tensors = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    tokenid_tensors = pad(tokenid_tensors, padding_value=0)
    # [batch_size, max_len, 1]
    tokenid_tensors = tokenid_tensors.unsqueeze(-1)
    if batch[0][2] is not None:
        # [batch_size]
        labelid_tensors = torch.tensor([item[2] for item in batch], dtype=torch.long)
    else:
        labelid_tensors = None
    
    return {"keys": keys, "tokenids": tokenid_tensors, "labelids": labelid_tensors, "parse_sents": parse_sents, "asr": asr, "gold_s": gold_s}