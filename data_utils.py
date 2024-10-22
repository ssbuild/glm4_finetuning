# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import copy
import glob
import json
import os
import typing
from functools import cache

import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, TrainingArgumentsAC
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import HfArgumentParser
from data_processer import DataStrategy, TokenIdsMaker
from deep_training.zoo.model_zoo.glm4.llm_model import ChatGLM4Tokenizer,PetlArguments,ChatGLMConfig
from config import *

assert config_args['max_seq_length'] > 20

data_conf = {
   'strategy': DataStrategy.truncation, # 数据策略选项
    DataStrategy.truncation: {
        'sup': True, # 是否监督训练
    },
    DataStrategy.siding: {
        'sliding_size': config_args['max_seq_length'] // 3 * 2, #prompt滑动窗口大小
        'sup': True, # 是否监督训练
        "src_max_length": config_args['max_seq_length'] - 10,
        "dst_max_length": None,
    },

}


def preprocess(text):
  #text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  # return text.replace("\\n", "\n").replace("\\t", "\t")
  return text

def build_masks_and_position_ids_glm(batch_input_ids, ctxlens):
    max_len = batch_input_ids.size(1)
    batch_position_ids, batch_attention_mask = [], []
    for input_ids,ctxlen in zip(batch_input_ids,ctxlens):
        position_ids = list(range(0,max_len))
        assert ctxlen <= max_len
        attention_mask = [1] * ctxlen + [0] * (max_len - ctxlen)
        batch_position_ids.append(torch.tensor(position_ids,dtype=torch.long))
        batch_attention_mask.append(torch.tensor(attention_mask,dtype=torch.long))

    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    batch_position_ids = torch.stack(batch_position_ids, dim=0)
    return batch_attention_mask,batch_position_ids

class NN_DataHelper(DataHelper):
    index = 1
    tokens_ids_maker = None
    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1


        tokenizer: ChatGLM4Tokenizer = self.tokenizer # noqa
        config: ChatGLMConfig = self.config           # noqa
        max_seq_length = self.max_seq_length_dict[mode]

        if self.tokens_ids_maker is None:
            self.tokens_ids_maker = TokenIdsMaker(tokenizer=tokenizer,config=config)


        examples = data

        strategy = data_conf['strategy']
        if strategy == DataStrategy.truncation:
            ds = self.tokens_ids_maker.trunction(tokenizer,config,examples=examples, max_seq_length=max_seq_length,**data_conf[strategy])
        elif strategy == DataStrategy.siding:
            ds = self.tokens_ids_maker.slidding(tokenizer,config, examples=examples, max_seq_length=max_seq_length, **data_conf[strategy])
        else:
            raise ValueError('Invalid strategy',strategy)

        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds



    def _get_messages(self, lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            conversations = jd['conversations']
            if line_id < 10:
                print(conversations)

            cid = 0
            sub = []
            while cid < len(conversations):
                m = conversations[cid]
                cid += 1
                role = m["from"]
                q = preprocess(m["value"])
                if role == "system":
                    assert len(sub) == 0
                    sub.append((role,q, m.pop('tools', None)))
                    continue
                assert role in ['user','observation']
                m = conversations[cid]
                cid += 1
                assert m["from"] == "assistant"
                a = preprocess(m["value"])
                assert len(a), ValueError('answer cannot empty')
                sub.append((role, q, a))
            D.append(sub)
        return D
    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        files = sum([glob.glob(file) for file in files], [])
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            D.extend(self._get_messages(lines))
        return D

    def collate_fn(self,batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        seqlens = o.pop('seqlen')
        max_len = torch.max(seqlens).tolist()
        input_ids = o['input_ids'][:, :max_len]

        attention_mask,position_ids = build_masks_and_position_ids_glm(input_ids,seqlens)
        o['input_ids'] = input_ids.long()
        o['attention_mask'] = attention_mask.long()
        o['position_ids'] = position_ids.long()
        o['labels'] = o['labels'][:, :max_len].long()
        return o

    def make_dataset_all(self):
        data_args = self.data_args

        # schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "labels": "int32_list",
            "seqlen": "int32_list",
        }
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True,
                                              mode='train',schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval',schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test',schema=schema)

        # 记录缓存文件
        with open(os.path.join(data_args.output_dir, 'intermediate_file_index.json'), mode='w',
                  encoding='utf-8') as f:
            f.write(json.dumps({
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "test_files": self.test_files,
            }, ensure_ascii=False))
    @cache
    def load_dataset_files(self):
        data_args = self.data_args
        if not data_args.convert_file:
            return {
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "test_files": self.test_files,
            }
        filename = os.path.join(data_args.output_dir, 'intermediate_file_index.json')
        assert os.path.exists(filename), 'make you dataset firstly'
        with open(filename, mode='r', encoding='utf-8') as f:
            return json.loads(f.read())

if __name__ == '__main__':
    if global_args[ "trainer_backend" ] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(config_args,
                                                                                         allow_extra_keys=True, )
    elif global_args[ "trainer_backend" ] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments))
        model_args, training_args, data_args, _ = parser.parse_dict(config_args)
    elif global_args["trainer_backend"] == "cl":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(config_args, allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(config_args,allow_extra_keys=True,)

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLM4Tokenizer,
                                                                  config_class_name=ChatGLMConfig)
    

    # 缓存数据集
    print(f'to make dataset is overwrite_cache {data_args.overwrite_cache}')
    dataHelper.make_dataset_all()

    print('make dataset complete!')
    print('check data !')
    dataset = dataHelper.load_sequential_sampler(dataHelper.load_dataset_files()["train_files"],
                                                 with_load_memory=data_args.data_backend == 'record',
                                                 batch_size=1,
                                                 collate_fn=dataHelper.collate_fn)

    print('total', len(dataset))
    for i, d in enumerate(dataset):
        print(d)
        if i > 3:
            break