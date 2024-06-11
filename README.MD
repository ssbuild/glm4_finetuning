### 
```text
    2024-06-10 initial glm4
```
   
##  statement
- [deep_training](https://github.com/ssbuild/deep_training)

## install
  - pip install -U -r requirements.txt
  - 如果无法安装 , 可以切换官方源 pip install -i https://pypi.org/simple -U -r requirements.txt  

```text
dev 通过一下方式安装
pip install -U git+https://github.com/ssbuild/deep_training.git
pip install -U transformers>=4.30 deepspeed xformers bitsandbytes>=0.39 accelerate>=0.20
```


## weight

- [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)   
- [glm-4-9b-chat-1m](https://huggingface.co/THUDM/glm-4-9b-chat-1m)  
- [glm-4-9b](https://huggingface.co/THUDM/glm-4-9b)  


## data sample
- open_data 不定时开放新数据 https://github.com/ssbuild/open_data


```json
{"id": 1, "conversations": [{"from": "system", "value": "You are ChatGLM4, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."}, {"from": "user", "value": "Hello"}, {"from": "assistant", "value": "Hello, I'm ChatGLM4. What can I assist you today?"}]}
{"id": 2, "conversations": [{"from": "system", "value": "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。", "tools": "[\n  {\n    \"type\": \"function\",\n    \"function\": {\n      \"name\": \"get_current_weather\",\n      \"description\": \"Get the current weather in a given location\",\n      \"parameters\": {\n        \"type\": \"object\",\n        \"properties\": {\n          \"location\": {\n            \"type\": \"string\",\n            \"description\": \"The city and state, e.g. San Francisco, CA\"\n          },\n          \"unit\": {\n            \"type\": \"string\"\n          }\n        },\n        \"required\": [\n          \"location\"\n        ]\n      }\n    }\n  }\n]"}, {"from": "user", "value": "今天北京的天气怎么样？"}, {"from": "assistant", "value": "{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"beijing\", unit=\"celsius\"}}\n"}, {"from": "observation", "value": "{\"temperature\": 22}"}, {"from": "assistant", "value": "根据查询结果，今天北京的气温为 22 摄氏度。"}]}
{"id": 3, "conversations": [{"from": "system", "value": "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。", "tools": "[\n  {\n    \"type\": \"function\",\n    \"function\": {\n      \"name\": \"get_recommended_books\",\n      \"description\": \"Get recommended books based on user's interests\",\n      \"parameters\": {\n        \"type\": \"object\",\n        \"properties\": {\n          \"interests\": {\n            \"type\": \"array\",\n            \"items\": {\n              \"type\": \"string\"\n            },\n            \"description\": \"The interests to recommend books for\"\n          }\n        },\n        \"required\": [\n          \"interests\"\n        ]\n      }\n    }\n  }\n]"}, {"from": "user", "value": "Hi, I am looking for some book recommendations. I am interested in history and science fiction."}, {"from": "assistant", "value": "{\"name\": \"get_recommended_books\", \"arguments\": {\"interests\": [\"history\", \"science fiction\"]}}\n"}, {"from": "observation", "value": "{\"books\": [\"Sapiens: A Brief History of Humankind by Yuval Noah Harari\", \"A Brief History of Time by Stephen Hawking\", \"Dune by Frank Herbert\", \"The Martian by Andy Weir\"]}"}, {"from": "assistant", "value": "Based on your interests in history and science fiction, I would recommend the following books: \"Sapiens: A Brief History of Humankind\" by Yuval Noah Harari, \"A Brief History of Time\" by Stephen Hawking, \"Dune\" by Frank Herbert, and \"The Martian\" by Andy Weir."}]}
{"id": 4, "conversations": [{"from": "user", "value": "写一个诗歌，关于冬天"}, {"from": "assistant", "value": "冬夜寂静冷，\n云在天边飘，\n冰封白雪上， \n寒冷像一场雪。\n \n雪花融化成冰，\n像那雪花飘洒，\n在寒冷的冬天，\n感受春天的喜悦。\n 冬日里，\n风雪渐消，\n一片寂静，\n把快乐和温暖带回家。"}]}
{"id": 5, "conversations": [{"from": "user", "value": "晚上睡不着应该怎么办"}, {"from": "assistant", "value": "晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:\n\n1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。\n2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。\n3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。\n4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。\n5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。\n6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。\n\n如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"}]}
```





## infer
    # infer.py 推理预训练模型
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理lora微调模型
     python infer.py







## training
```text
# 制作数据
cd scripts
bash train_full.sh -m dataset 
or
bash train_lora.sh -m dataset 
or
bash train_ptv2.sh -m dataset 

注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)

# 全参数训练 
    bash train_full.sh -m train 
    
# lora adalora ia3 
    bash train_lora.sh -m train 
    
# ptv2
    bash train_ptv2.sh -m train 
```

## 训练参数
[训练参数](assets/args.MD)




## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [moss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
- [chatglm3_finetuning](https://github.com/ssbuild/chatglm3_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)
- [xverse_finetuning](https://github.com/ssbuild/xverse_finetuning)
- [internlm_finetuning](https://github.com/ssbuild/internlm_finetuning)
- [qwen_finetuning](https://github.com/ssbuild/qwen_finetuning)
- [skywork_finetuning](https://github.com/ssbuild/skywork_finetuning)
- [bluelm_finetuning](https://github.com/ssbuild/bluelm_finetuning)
- [yi_finetuning](https://github.com/ssbuild/yi_finetuning)
- [glm4_finetuning](https://github.com/ssbuild/glm4_finetuning)
- [aigc_evals](https://github.com/ssbuild/aigc_evals)
- [aigc_serving](https://github.com/ssbuild/aigc_serving)


## 
    纯粹而干净的代码


    

## Reference
    https://huggingface.co/THUDM/glm4




## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/glm4_finetuning&type=Date)](https://star-history.com/#ssbuild/glm4_finetuning&Date)

