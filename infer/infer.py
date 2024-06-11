# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper
from deep_training.zoo.model_zoo.glm4.llm_model import MyTransformer,ChatGLM4Tokenizer,PetlArguments,setup_model_profile, ChatGLMConfig
from deep_training.zoo.model_zoo.glm4.llm_model import RotaryNtkScaledArguments,RotaryLinearScaledArguments # aigc-zoo 0.1.20


if __name__ == '__main__':
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args,allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLM4Tokenizer
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLM4Tokenizer, config_class_name=ChatGLMConfig)


    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16)

    model = pl_model.get_llm_model()
    # 已经量化
    model.half().cuda()
    model = model.eval()

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    for input in text_list:
        response, history = model.chat(tokenizer, input, history=[], max_length=2048,
                                       do_sample=True, top_p=0.8, temperature=0.8, )
        print("input", input)
        print("response", response)

    # response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=30)
    # print('写一个诗歌，关于冬天',' ',response)

