# Chinese Spoken Named Entity Recognition in Real-world Scenarios: Dataset and Approaches
This is the repo for the paper "Chinese Spoken Named Entity Recognition in Real-world Scenarios: Dataset and Approaches". The paper is under review. This is an anonymous repo. 

## Abstract
Spoken Named Entity Recognition (NER) aims to extract entities from speech. 
The extracted entities can help voice assistants better understand user's questions and instructions.
However, current Chinese Spoken NER datasets are laboratory-controlled data that collected by reading existing texts in quiet environments, rather than natural spoken data, and the texts used for reading are also limited in topics. 
These limitations obstruct the development of Spoken NER in more natural and common real-world scenarios.
To address this gap, we introduce a 
real-world Chinese Spoken NER dataset (RWCS-NER), encompassing open-domain daily conversations and task-oriented intelligent cockpit instructions. 
We compare several mainstream pipeline approaches on RWCS-NER. The results indicate that the current methods, affected by Automatic Speech Recognition (ASR) errors, do not perform satisfactorily in real settings.
Aiming to enhance Spoken NER in real-world scenarios, we propose two approaches: self-training-asr and mapping then distilling (MDistilling).
Experiments show that both approaches can achieve significant improvements, particularly MDistilling.
Even compared with GPT4.0, MDistilling still reaches better results.
We believe that our work will advance the field of Spoken NER in real-world settings.

## Installation
```shell
pip install -r requirements.txt
```
## Data
***Our dataset (DC and ICI), consisting of audio, text, and NER annotations, is open source for the community.***
The annotated data is in the `data_to_upload` folder. The data is in jsonl format.
The folder is divided into four parts: MSRA, DC, ICI and audios.
The MSRA data is from the MSRA-NER dataset, which is widely used in Chinese NER tasks.
DC and ICI are our real-world Chinese Spoken NER datasets.
The audios are the corresponding speech records of DC and ICI.
Considering the size of the data, we only uploaded some sample audios instead of all of them (the same goes for DC's training set). 
But we have uploaded the complete annotations for the development set and test set.
We will provide download links for all the audios and DC train data in future public versions.

## Inference with LLMs
```shell
## take DC Test as an example
### on transcibed asr data with qwen1.5-14B-chat
python runllm.py predict --input_jsonl_file data_to_upload/dc/test_example.jsonl \
        --model Qwen/Qwen1.5-14B-Chat \
        --shot 1 \
        --onasr \
        --method QA

### on gold data with qwen1.5-14B-chat
python runllm.py predict --input_jsonl_file data_to_upload/dc/test_example.jsonl \
        --model Qwen/Qwen1.5-14B-Chat \
        --shot 1 \
        --method QA

### on gold data with gpt4, should first set the openai api key in the environment variable OPENAI_API_KEY
python rungptner.py predict --input_jsonl_file data_to_upload/dc/test_example.jsonl \
        --shot 1 \
        --model gpt-4-1106-preview \
        --method QA 

### note that if run on ici data, should set --icsr, for example
python runllm.py predict --input_jsonl_file data_to_upload/ici/test_example_10.jsonl \
        --model Qwen/Qwen1.5-14B-Chat \
        --shot 1 \
        --onasr \
        --icsr \
        --method QA
```

## Baseline
* Train a msra roberta-crf-based baseline model
```shell
python -m main train --path exp/msra-seed8-epo20 \
                     --roberta_path chinese-roberta-wwm-ext-large \
                     --train_file data_to_upload/MSRA/train.jsonl \
                     --dev_file data_to_upload/MSRA/dev.jsonl \
                     --test_file data_to_upload/MSRA/test.jsonl \
                     --batch_size 64 \
                     --epochs 20 \
                     --seed 8 \
                     --device 0
```
* evaluate and predict
```shell
python -m main evaluate --input_file data_to_upload/MSRA/dev.jsonl \
        --path exp/msra-seed8-epo20/best.model \
        --device 0

python -m main predict --input_file data_to_upload/MSRA/dev.jsonl \
        --output_file baseline-msra-dev-pred.jsonl \
        --path exp/msra-seed8-epo20/best.model \
        --device 0
```

## Self-training-gold
```shell
python -m main self-train --select_file_dir st-ici-dropnull05 \
                     --path exp/st-ici-dropnull05 \
                     --icsr \
                     --unlabel_file data_to_upload/ici/unlabel_train.jsonl \
                     --src_train_file data_to_upload/MSRA/train.jsonl \
                     --init_model_path exp/msra-seed8-epo20/best.model \
                     --dev_file data_to_upload/ici/dev_example_10.jsonl \
                     --test_file data_to_upload/ici/test_example_10.jsonl \
                     --batch_size 64 \
                     --p_drop 0.5 \
                     --p_hold 0.9 \
                     --device 0
```

## Self-training-asr
```shell
python -m main self-train --select_file_dir stasr-ici-dropnull05 \
                     --path exp/stasr-ici-dropnull05 \
                     --icsr \
                     --unlabel_file data_to_upload/ici/unlabel_train_asrtxt.jsonl \
                     --src_train_file data_to_upload/MSRA/train.jsonl \
                     --init_model_path exp/msra-seed8-epo20/best.model \
                     --dev_file data_to_upload/ici/dev_example_10.jsonl \
                     --test_file data_to_upload/ici/test_example_10.jsonl \
                     --batch_size 64 \
                     --p_drop 0.5 \
                     --p_hold 0.9 \
                     --device 0
```

## Mapping then distilling
```shell
# 1. use teacher model to predict on gold unlabel data
python -m main predict --input_file data_to_upload/ici/unlabel_train.jsonl \
        --output_file ici-tea-gold-pred.jsonl \
        --path exp/st-ici-dropnull05/best.model \
        --device 0

# 2. map the predicted label to the unlabel asr data
python process_json_file.py --src_file data_to_upload/MSRA/train.jsonl \
        --tea_pred_file ici-tea-gold-pred.jsonl \
        --asr_file data_to_upload/ici/unlabel_train_asrtxt.jsonl \
        --tgt_file ici-MDistill.jsonl

# 3. train with the MDistill data
python -m main train --path exp/MDistill-ici-dropnull05 \
                     --roberta_path chinese-roberta-wwm-ext-large \
                     --train_file ici-MDistill.jsonl \
                     --dev_file data_to_upload/ici/dev_example_10.jsonl \
                     --test_file data_to_upload/ici/test_example_10.jsonl \
                     --batch_size 64 \
                     --lr 5e-4 \
                     --epochs 20 \
                     --seed 8 \
                     --device 0

```
