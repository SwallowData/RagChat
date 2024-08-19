# !/user/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd
from langchain.memory import ConversationBufferMemory
class ChatMemory():
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    FILE_PATH = '../data/history_data.csv'
    @classmethod
    def init_csv(cls):
        df_init = pd.DataFrame([['input', 'output']])
        df_init.to_csv(cls.FILE_PATH, mode='w', header=False, index=False)
    if not os.path.exists(FILE_PATH):
        init_csv()
    @classmethod
    def add_chat_history(cls,input=None,output=None):
        print('FILE_PATH:',cls.FILE_PATH)
        if input:
            df = pd.DataFrame([[input,output]],columns=["input",'output'])
            # print(df)
            df.to_csv(cls.FILE_PATH,mode = 'a', header=False,index=False)
            print('This chat was successfully saved')

    def read_chat_from_csv(self,FILE_PATH=FILE_PATH) -> list:
        try:
            df = pd.read_csv(FILE_PATH)
        except Exception as e:
            df = None
            print(e)
            return
        # print(df.values.tolist())
        for i in df.values.tolist():
            self.memory.save_context({"input": i[0]}, {"ouput": i[1]})
        print('success read chat data save the memory')