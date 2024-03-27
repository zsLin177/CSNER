import openai
import json
import argparse
import tqdm
from myscripts.datascripts.generate_super_text import PER_left, PER_right, LOC_left, LOC_right, ORG_left, ORG_right
import os

openai.api_key = "your_key_here"

def read_jsonl(file):
    with open(file, "r", encoding="utf-8") as reader:
        res = []
        for line in reader:
            res.append(json.loads(line))
        return res
    

def generate_messages(input_sent, examples, shot=3):
    messages = []
    init_prompt =  f"你是一位优秀的语言学家。你的任务是在给定的句子中标注人名person实体、地名location实体，以及组织organization实体。人名实体用{PER_left} {PER_right}标注，地名实体用{LOC_left} {LOC_right}标注，组织实体用{ORG_left} {ORG_right}标注。"
    messages.append({"role": "system", "content": init_prompt})
    for example in examples[0:shot]:
        if "sentences" in example:
            raw_sent = example['sentences']
        elif "text" in example:
            raw_sent = example['text']
        else:
            raise ValueError("no text found")
        messages.append({"role": "user", "content": raw_sent})
        messages.append({"role": "assistant", "content": example['super_text']})
    messages.append({"role": "user", "content": input_sent})

    return messages

def generate_QA_messages(input_sent, examples, shot=3):
    messages = []
    init_prompt =  f"你是一位优秀的语言学家。你的任务是识别给定的句子中的人person实体、地点location实体、组织organization实体。按照以下格式返回答案：person: \nlocation: \norganization: 。每个实体之间以空格间隔，若没有对应类型的实体，请返回空字符串。不要输出额外的内容。"
    messages.append({"role": "system", "content": init_prompt})
    for example in examples[0:shot]:
        if "sentences" in example:
            raw_sent = example['sentences']
        elif "text" in example:
            raw_sent = example['text']
        else:
            raise ValueError("no text found")
        messages.append({"role": "user", "content": raw_sent})

        # form answer
        if len(example["label"]) == 0:
            answer = "person: \nlocation: \norganization: "
        else:
            per_s_set = []
            loc_s_set = []
            org_s_set = []
            for ne in example["label"]:
                if ne[2] == "PER" and ne[3] not in per_s_set:
                    per_s_set.append(ne[3])
                elif ne[2] == "LOC" and ne[3] not in loc_s_set:
                    loc_s_set.append(ne[3])
                elif ne[2] == "ORG" and ne[3] not in org_s_set:
                    org_s_set.append(ne[3])
                elif ne[2] not in ["PER", "LOC", "ORG"]:
                    raise ValueError("ne type error")
                    
            per_s = " ".join(per_s_set) if len(per_s_set) > 0 else "None"
            loc_s = " ".join(loc_s_set) if len(loc_s_set) > 0 else "None"
            org_s = " ".join(org_s_set) if len(org_s_set) > 0 else "None"
            answer = f"person: {per_s}\nlocation: {loc_s}\norganization: {org_s}"
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": input_sent})

    return messages
    
def process(input_jsonl_file, model, method, shot=3, onasr=False, icsr=False):
    print(f"using model {model}, shot {shot}, method {method}")
    data = read_jsonl(input_jsonl_file)
    if not onasr:
        output_jsonl_file = input_jsonl_file.replace(".jsonl", f"-{model}-shot{shot}-{method}.jsonl")
    else:
        output_jsonl_file = input_jsonl_file.replace(".jsonl", f"-asr-{model}-shot{shot}-{method}.jsonl")
    if os.path.exists(output_jsonl_file):
        exist_data = read_jsonl(output_jsonl_file)
        print(f"exist processed data {len(exist_data)}")
        exist_keys = set([d['key'] for d in exist_data])
        new_data = [d for d in data if d['key'] not in exist_keys]
        print(f"remaining data {len(new_data)}")
    else:
        new_data = data
    with open(output_jsonl_file, "a", encoding="utf-8") as writer:
        for this_dic in tqdm.tqdm(new_data):
            try:
                examples = this_dic['examples']
            except:
                assert shot == 0
                examples = []
            if not onasr:
                if "sentences" in this_dic:
                    input_text = this_dic['sentences']
                elif "text" in this_dic:
                    input_text = this_dic['text']
                else:
                    raise ValueError("no text found")
            else:
                input_text = this_dic['asr']
            if method == "special_token":
                messages = generate_messages(input_text, examples, shot)
            elif method == "QA":
                messages = generate_QA_messages(input_text, examples, shot)
            flag = False
            while not flag:
                try:
                    response = openai.ChatCompletion.create(model=model, messages=messages, seed=1024, temperature=0.8)
                    flag = True
                except KeyboardInterrupt:
                    exit()
                except Exception as e:
                    print(f"An error occurred: {e}, retrying...")
            this_dic['gpt_res'] = response['choices'][0]['message']['content']
            writer.write(json.dumps(this_dic, ensure_ascii=False)+'\n')
            writer.flush()
    print(f"output file {output_jsonl_file} generated")
    evaluate_QA(output_jsonl_file, onasr=onasr, icsr=icsr, result_file=output_jsonl_file.replace(".jsonl", ".prf"))

def evaluate_QA(input_jsonl_file, onasr=False, icsr=False, result_file=None):
    data = read_jsonl(input_jsonl_file)
    gold_num = .0
    gold_per_num = .0
    gold_loc_num = .0
    gold_org_num = .0

    pred_num = .0
    pred_per_num = .0
    pred_loc_num = .0
    pred_org_num = .0

    correct_num = .0
    correct_per_num = .0
    correct_loc_num = .0
    correct_org_num = .0

    if result_file is not None:
        output_f = open(result_file, "w", encoding="utf-8")
    else:
        output_f = open(input_jsonl_file.replace(".jsonl", ".prf"), "w", encoding="utf-8")
    
    if onasr:
        for this_dic in data:
            if icsr:
                for oth_label in this_dic['other_label']:
                    if oth_label[3] == "#":
                        ins_label = oth_label[2]
                        break
            else:
                ins_label = None

            gold_lst = [tuple(ne_lst[2:]) for ne_lst in this_dic['label']]
            gpt_res = this_dic['gpt_res']
            try:
                raw_text = this_dic['asr']
            except:
                # raise KeyError
                raw_text = this_dic['text']
            pred_lst = extract_res_from_answer(gpt_res, raw_text, onasr, ins_label)
            gold_num += len(gold_lst)
            pred_num += len(pred_lst)
            for pred in pred_lst:
                if pred in gold_lst:
                    correct_num += 1
                    gold_lst.remove(pred)
        precision = correct_num / pred_num
        recall = correct_num / gold_num
        f1 = 2 * precision * recall / (precision + recall)
        print(f"gold_num: {gold_num}, pred_num: {pred_num}, correct_num: {correct_num}")
        print(f"precision: {precision:6.2%}, recall: {recall:6.2%}, f1: {f1:6.2%}")

        output_f.write(f"gold_num: {gold_num}, pred_num: {pred_num}, correct_num: {correct_num}\n")
        output_f.write(f"precision: {precision:6.2%}, recall: {recall:6.2%}, f1: {f1:6.2%}\n")
        return

    for this_dic in data:
        if icsr:
            for oth_label in this_dic['other_label']:
                if oth_label[3] == "#":
                    ins_label = oth_label[2]
                    break
        else:
            ins_label = None

        gold_set = set()
        gold_per_set = set()
        gold_loc_set = set()
        gold_org_set = set()

        pred_set = set()
        pred_per_set = set()
        pred_loc_set = set()
        pred_org_set = set()

        for ne_lst in this_dic['label']:
            gold_set.add(tuple(ne_lst))
            if ne_lst[2] == 'PER':
                gold_per_set.add(tuple(ne_lst))
            elif ne_lst[2] == 'LOC':
                gold_loc_set.add(tuple(ne_lst))
            elif ne_lst[2] == 'ORG':
                gold_org_set.add(tuple(ne_lst))
            else:
                raise ValueError(f"ne_type {ne_lst[2]} not supported")
        gpt_res = this_dic['gpt_res']

        if "sentences" in this_dic:
            raw_text = this_dic['sentences']
        elif "text" in this_dic:
            raw_text = this_dic['text']
        else:
            raise ValueError("no text found")
        ne_lsts = extract_res_from_answer(gpt_res, raw_text, ins_label=ins_label)
        for ne_lst in ne_lsts:
            pred_set.add(ne_lst)
            if ne_lst[2] == 'PER':
                pred_per_set.add(ne_lst)
            elif ne_lst[2] == 'LOC':
                pred_loc_set.add(ne_lst)
            elif ne_lst[2] == 'ORG':
                pred_org_set.add(ne_lst)
            else:
                raise ValueError(f"ne_type {ne_lst[2]} not supported")
        gold_num += len(gold_set)
        gold_per_num += len(gold_per_set)
        gold_loc_num += len(gold_loc_set)
        gold_org_num += len(gold_org_set)

        pred_num += len(pred_set)
        pred_per_num += len(pred_per_set)
        pred_loc_num += len(pred_loc_set)
        pred_org_num += len(pred_org_set)

        correct_num += len(gold_set & pred_set)
        correct_per_num += len(gold_per_set & pred_per_set)
        correct_loc_num += len(gold_loc_set & pred_loc_set)
        correct_org_num += len(gold_org_set & pred_org_set)
    
    precision = correct_num / pred_num
    recall = correct_num / gold_num
    f1 = 2 * precision * recall / (precision + recall)

    precision_per = correct_per_num / pred_per_num
    recall_per = correct_per_num / gold_per_num
    f1_per = 2 * precision_per * recall_per / (precision_per + recall_per)

    precision_loc = correct_loc_num / pred_loc_num
    recall_loc = correct_loc_num / gold_loc_num
    f1_loc = 2 * precision_loc * recall_loc / (precision_loc + recall_loc)

    precision_org = correct_org_num / pred_org_num
    recall_org = correct_org_num / gold_org_num
    f1_org = 2 * precision_org * recall_org / (precision_org + recall_org)

    print(f"gold_num: {gold_num}, pred_num: {pred_num}, correct_num: {correct_num}")
    print(f"precision: {precision:6.2%}, recall: {recall:6.2%}, f1: {f1:6.2%}")
    print(f"gold_per_num: {gold_per_num}, pred_per_num: {pred_per_num}")
    print(f"gold_loc_num: {gold_loc_num}, pred_loc_num: {pred_loc_num}")
    print(f"gold_org_num: {gold_org_num}, pred_org_num: {pred_org_num}")
    print(f"precision_per: {precision_per:6.2%}, recall_per: {recall_per:6.2%}, f1_per: {f1_per:6.2%}")
    print(f"precision_loc: {precision_loc:6.2%}, recall_loc: {recall_loc:6.2%}, f1_loc: {f1_loc:6.2%}")
    print(f"precision_org: {precision_org:6.2%}, recall_org: {recall_org:6.2%}, f1_org: {f1_org:6.2%}")

    output_f.write(f"gold_num: {gold_num}, pred_num: {pred_num}, correct_num: {correct_num}\n")
    output_f.write(f"precision: {precision:6.2%}, recall: {recall:6.2%}, f1: {f1:6.2%}\n")
    output_f.write(f"gold_per_num: {gold_per_num}, pred_per_num: {pred_per_num}\n")
    output_f.write(f"gold_loc_num: {gold_loc_num}, pred_loc_num: {pred_loc_num}\n")
    output_f.write(f"gold_org_num: {gold_org_num}, pred_org_num: {pred_org_num}\n")
    output_f.write(f"precision_per: {precision_per:6.2%}, recall_per: {recall_per:6.2%}, f1_per: {f1_per:6.2%}\n")
    output_f.write(f"precision_loc: {precision_loc:6.2%}, recall_loc: {recall_loc:6.2%}, f1_loc: {f1_loc:6.2%}\n")
    output_f.write(f"precision_org: {precision_org:6.2%}, recall_org: {recall_org:6.2%}, f1_org: {f1_org:6.2%}\n")
    output_f.close()

def extract_res_from_answer(answer, raw_text, onasr=False, ins_label=None):
    try:
        assert len(answer.split("\n")) == 3
        # per_s, loc_s, org_s = answer.split("\n")
        per_s, loc_s, org_s = "", "", ""
        for s in answer.split("\n"):
            if s.startswith("person"):
                per_s = s
            elif s.startswith("location"):
                loc_s = s
            elif s.startswith("organization"):
                org_s = s
            else:
                continue
    except:
        return []
    ne_dict = {}

    if per_s != "":
        per_s = per_s.split(":")[1].strip()
        per_s_set = set(per_s.split(" "))
        for per in per_s_set:
            if per not in [" ", "", "None"]:
                ne_dict[per] = "PER"
    if loc_s != "":
        loc_s = loc_s.split(":")[1].strip()
        loc_s_set = set(loc_s.split(" "))
        for loc in loc_s_set:
            if loc not in [" ", "", "None"]:
                ne_dict[loc] = "LOC"
    if org_s != "":
        org_s = org_s.split(":")[1].strip()
        org_s_set = set(org_s.split(" "))
        for org in org_s_set:
            if org not in [" ", "", "None"]:
                if ins_label == "导航":
                    ne_dict[org] = "LOC"
                else:
                    ne_dict[org] = "ORG"
    
    if len(ne_dict) == 0:
        return []

    # use max match to find the start and end position of each ne
    ne_lsts = []
    max_len = min(max([len(ne_s) for ne_s in ne_dict]), len(raw_text))
    start = 0
    flag = False
    while start < len(raw_text):
        for end in range(min(start+max_len, len(raw_text)), start, -1):
            if raw_text[start:end] in ne_dict:
                if onasr:
                    ne_lsts.append((ne_dict[raw_text[start:end]], raw_text[start:end]))
                else:
                    ne_lsts.append((start, end, ne_dict[raw_text[start:end]], raw_text[start:end]))
                start = end
                flag = True
                break
        if not flag:
            start += 1
        flag = False
    return ne_lsts

def evaluate(input_jsonl_file, method, onasr=False, icsr=False):
    if method == "QA":
        evaluate_QA(input_jsonl_file, onasr, icsr)


if __name__ == "__main__":
    # examples = read_examples("example_by_speak2normal.txt")
    # normal_lst = get_normal_input("same_ext152.jsonl")
    # messages = generate_messages(examples, normal_lst)
    # responses = get_gpt_response(messages, outputfile="gpt_response_ext152.jsonl")


    # this_message = [{"role": "system", "content": "你是一位优秀的语言学家。你的任务是在给定的句子中标注人名person实体、地名location实体，以及组织organization实体。人名实体用[[ ]]标注，地名实体用<< >>标注，组织实体用(( ))标注。"}, 
    # {"role": "user", "content": "但是苹果苹果手机是吧"},
    # {"role": "assistant", "content": "但是((苹果))((苹果))手机是吧"},
    # {"role": "user", "content": "苹果的市场做得好大你知不知道"},
    # {"role": "assistant", "content": "((苹果))的市场做得好大你知不知道"},
    # {"role": "user", "content": "其实中国手机我一直想说的一点就是有点模仿嗯缺乏创新模仿苹果"},
    # {"role": "assistant", "content": "其实<<中国>>手机我一直想说的一点就是有点模仿嗯缺乏创新模仿((苹果))"},
    # {"role": "user", "content": "苹果是在些滴滴方面收费也比安卓贵就这点很不好但苹果的确好使"},
    # ]
    # response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", messages=this_message, seed=1024, temperature=1.0)
    # # print(this_message)
    # print(response['choices'][0]['message']['content'])

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl_file", type=str)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--shot", type=int, default=3)
    args = parser.parse_args()
    process(args.input_jsonl_file, args.model, args.shot)

