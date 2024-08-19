# !/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from memory_model import ChatMemory
@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    print('temp_dir.name:',temp_dir.name)
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        #将文件写入临时文件夹
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        #将数据加载入docs
        docs.extend(loader.load())
    #chunk_size 划分的尺寸  chunk_overlap 两个文档之间可以重复的字符数量
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    print('splits:',splits)
    # Create embeddings and store in vectordbs
    embeddings = ZhipuAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory='../data/chroma{temp_dir.name}'  # 允许我们将persist_directory目录保存到磁盘上
    )
    print('vectordb:',vectordb)
    ChatMemory.init_csv()
    return vectordb