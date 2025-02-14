# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper, global_args
from deep_training.zoo.model_zoo.glm4.llm_model import MyTransformer, ChatGLM4Tokenizer, setup_model_profile, ChatGLMConfig, \
    PetlArguments,PetlModel

if __name__ == '__main__':
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args, ) = parser.parse_dict(config_args, allow_extra_keys=True)
    setup_model_profile()
    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLM4Tokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLM4Tokenizer, config_class_name=ChatGLMConfig)

    ckpt_dir = './best_ckpt/last'
    config = ChatGLMConfig.from_pretrained(ckpt_dir)

    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True and config.pre_seq_len is None

    # new_num_tokens = config.vocab_size
    # if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
    #     config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,
                             # new_num_tokens=new_num_tokens,#扩充词
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )
    # 加载多个lora权重
    pl_model.load_sft_weight(ckpt_dir,adapter_name="default")

    #加载多个lora权重
    pl_model.load_sft_weight(ckpt_dir, adapter_name="default2")




    pl_model.eval().half().cuda()

    # backbone model replaced PetlModel
    lora_model: PetlModel = pl_model.backbone

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]

    # 基准模型推理
    with lora_model.disable_adapter():
        for input in text_list:
            #lora_model 调用子对象方法
            response, history = lora_model.chat(tokenizer, input, history=[], max_length=2048,
                                           do_sample=True, top_p=0.8, temperature=0.8, )
            print("input", input)
            print("response", response)

    lora_model.set_adapter(adapter_name='default')

    for input in text_list:
        # lora_model 调用子对象方法
        response, history = lora_model.chat(tokenizer, input, history=[], max_length=2048,
                                            do_sample=True, top_p=0.8, temperature=0.8, )
        print("input", input)
        print("response", response)

