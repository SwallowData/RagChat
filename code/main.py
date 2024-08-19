# !/user/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
from chat_return import generate_response,get_chat_qa_chain,get_qa_chain

if __name__ == '__main__':
    st.title('🦜🔗 动手学大模型应用开发')
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
    openai_api_base = st.sidebar.text_input('OpenAI API Base', type='password')
    if not (openai_api_base and openai_api_base):
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload PDF documents to continue.")
        st.stop()

    # 添加一个选择按钮来选择不同的模型
    # selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions=["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=500)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt, openai_api_key, openai_api_base)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt, openai_api_key, openai_api_base,uploaded_files)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(question=prompt, openai_api_key=openai_api_key, openai_api_base=openai_api_base,
                                       uploaded_files=uploaded_files)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])