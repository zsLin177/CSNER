import openai
import torch
import json
import tqdm

openai.api_key = "your-api-key"

def store_text_embedding(text, model='text-embedding-ada-002'):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def save_text_embedding(jsonl_file, path, model='text-embedding-ada-002'):
    with open(jsonl_file, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f]
    if 'key' in data[0]:
        keys = [dic['key'] for dic in data]
    else:
        keys = [i for i in range(len(data))]
    try:
        texts = [dic['text'] for dic in data]
    except:
        texts = [dic['sentence'] for dic in data]
    f = open(path+".str", 'w')
    embeddings = []
    for text in tqdm.tqdm(texts):
        flag = False
        tried = 0
        while not flag:
            tried += 1
            try:
                this_embed = store_text_embedding(text, model)
                f.write(str(this_embed)+'\n')
                f.flush()
                embeddings.append(this_embed)
                flag = True
                print('tried: ', tried, end='')
            except:
                pass
    
    f.close()
    state_dict = {'keys': keys, 'embeddings': embeddings}
    torch.save(state_dict, path)

def load_text_embedding(path):
    state_dict = torch.load(path)
    keys = state_dict['keys']
    embeddings = state_dict['embeddings']
    embed_tensor = torch.tensor(embeddings)
    return keys, embed_tensor

def get_topk_similar(q_tensors, k_tensors, k=5):
    "use the dot product to get the similarity between q and k "
    sim = torch.matmul(q_tensors, k_tensors.transpose(0, 1))
    values, indices = sim.topk(k, dim=1)
    return values, indices

if __name__ == '__main__':
    save_text_embedding('MSRA/train.json', 'MSRA/train_text_embedding.pt')
    # keys, embeds = load_text_embedding('ramc/dev_text_embedding.pt')
    # values, indices = get_topk_similar(embeds, embeds, k=5)