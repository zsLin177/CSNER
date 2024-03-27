import openai
import json
import random

openai.api_key = "xxx"

def read_examples(input_file):
    with open(input_file, "r", encoding="utf-8") as reader:
        res = []
        for line in reader:
            speak, normal = line.strip().split(" ")
            res.append({'speak': speak, 'normal': normal})
        return res
    
def get_normal_input(file):
    with open(file, "r", encoding="utf-8") as reader:
        res = []
        for line in reader:
            res.append(line.strip())
        return res

def generate_messages(examples, normal_lst, shot=3):
    messages = []
    prompt = "帮我把我输入的书写文本，转换为口语风格，比如添加语气词、口吃等口语现象。添加口吃的话，不要加太多："
    for normal_input in normal_lst:
        this_message = [{"role": "system", "content": "You are a helpful assistant."},]
        context_example = random.sample(examples, shot)
        for example in context_example:
            this_message.append({"role": "user", "content": prompt + example['normal']})
            this_message.append({"role": "assistant", "content": example['speak']})
        this_message.append({"role": "user", "content": prompt + normal_input})
        messages.append(this_message)

    return messages

def get_gpt_response(messages, model="gpt-3.5-turbo-0613", outputfile="gpt_response.jsonl"):
    responses = []
    for message in messages:
        response = openai.ChatCompletion.create(
        model=model,
        messages=message,
        )
        responses.append(response)
        print(message)
        print(response['choices'][0]['message']['content'])
    
    with open(outputfile, "w", encoding="utf-8") as writer:
        for response in responses:
            writer.write(json.dumps(response) + "\n")
    return responses

if __name__ == "__main__":
    examples = read_examples("../example_by_speak2normal.txt")
    normal_lst = get_normal_input("../example_by_normal2speak.txt")
    messages = generate_messages(examples, normal_lst)
    responses = get_gpt_response(messages)

        

    