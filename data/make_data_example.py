# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 12:50


import json

# 对话例子
x0 = [
    {
        "role": "system",
        "q": "You are ChatGLM4, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.",
        "a": ""
    },
    {
        "role": "user",
        "q": "Hello",
        "a": "Hello, I'm ChatGLM4. What can I assist you today?"
    }
]


#工具例子

x1 = [
    {
        "role": "system",
        "q": "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。",
        "tools":
[
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string"},
                },
                "required": ["location"],
            },
        }
    }
],
        "a": ""
    },
    {
        "role": "user",
        "q": "今天北京的天气怎么样？",
        "a": """{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"beijing\", unit=\"celsius\"}}
"""},
    {
        "role": "observation",
        "q": '{"temperature": 22}',
        "a": "根据查询结果，今天北京的气温为 22 摄氏度。"
    }
]


x2 = [
    {
        "role": "system",
        "q": "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。",
        "tools":
[
    {
        "type": "function",
        "function": {
            "name": "get_recommended_books",
            "description": "Get recommended books based on user's interests",
            "parameters": {
                "type": "object",
                "properties": {
                    "interests": {
                        "type": "array",
                        "items": {
                        "type": "string"
                    },
                    "description": "The interests to recommend books for"
                    }
                },
                "required": [
                    "interests"
                ]
            }
        }
    }
],
        "a": ""
    },
    {
        "role": "user",
        "q": "Hi, I am looking for some book recommendations. I am interested in history and science fiction.",
        "a": """{\"name\": \"get_recommended_books\", \"arguments\": {\"interests\": [\"history\", \"science fiction\"]}}
"""},
    {
        "role": "observation",
        "q": "{\"books\": [\"Sapiens: A Brief History of Humankind by Yuval Noah Harari\", \"A Brief History of Time by Stephen Hawking\", \"Dune by Frank Herbert\", \"The Martian by Andy Weir\"]}",
        "a": "Based on your interests in history and science fiction, I would recommend the following books: \"Sapiens: A Brief History of Humankind\" by Yuval Noah Harari, \"A Brief History of Time\" by Stephen Hawking, \"Dune\" by Frank Herbert, and \"The Martian\" by Andy Weir."
    }
]


x3 = [
    {
        "role": "user",
        "q": "写一个诗歌，关于冬天",
        "a": '\n'.join([
            "冬夜寂静冷，",
            "云在天边飘，", "冰封白雪上， ", "寒冷像一场雪。",
            " ",
            "雪花融化成冰，",
            "像那雪花飘洒，",
            "在寒冷的冬天，",
            "感受春天的喜悦。",
            " 冬日里，",
            "风雪渐消，",
            "一片寂静，",
            "把快乐和温暖带回家。"
        ])

    }
]


x4 = [
    {
        "role": "user",
        "q": "晚上睡不着应该怎么办",
        "a": '\n'.join([
            "晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:",
            "",
            "1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。",
            "2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。",
            "3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。",
            "4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。",
            "5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。",
            "6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。",
            "",
            "如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"
        ])
    }

]





x = [x0,x1,x2,x3,x4]


with open('./finetune_train_conversations.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for i in range(50):
        for j in range(len(x)):
            index += 1

            conversation = []
            for item in x[j]:
                role = item.get("role","user")
                if role == "system":
                    conversation.append( {
                        "from":  item.get("role","user"),
                        "value": item["q"]
                    })
                    if 'tools' in item:
                        conversation[-1]["tools"] = json.dumps(item["tools"],ensure_ascii=False,indent=2)

                else:
                    conversation.append({
                        "from": item.get("role", "user"),
                        "value": item["q"]
                    })
                    conversation.append({
                        "from": "assistant",
                        "value": item["a"]
                    })

            conversations = {
                "id": index,
                "conversations": conversation
            }
            f.write(json.dumps(conversations,ensure_ascii=False) + '\n' )