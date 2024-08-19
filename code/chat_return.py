# !/user/bin/env python3
# -*- coding: utf-8 -*-
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from memory_model import ChatMemory
from become_vector import configure_retriever
#初始化记忆功能
Memory= ChatMemory()
print('Memory.memory:',Memory.memory)
print('type(Memory.memory):',type(Memory.memory))
#普通问答
def generate_response(input_text, openai_api_key,openai_api_base):
    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key,base_url=openai_api_base)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output)
    return output

#带有历史记录的问答链
def get_chat_qa_chain(question:str,openai_api_key:str,openai_api_base:str,uploaded_files):
    vectordb = configure_retriever(uploaded_files)
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0.5,openai_api_key = openai_api_key,base_url = openai_api_base)
    Memory.read_chat_from_csv()
    #定义检索器
    retriever=vectordb.as_retriever()
    print("get_chat_qa_chain:retriever-", retriever)
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=Memory.memory
    )
    print("memory before query:", Memory.memory)
    result = qa({"question": question})
    # memory.save_context({"input": question}, {"ouput": result['answer']})
    Memory.add_chat_history(input=question,output=result['answer'])
    print("memory after query:",Memory.memory)
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question:str,openai_api_key:str,openai_api_base,uploaded_files):
    vectordb = configure_retriever(uploaded_files)
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,openai_api_key = openai_api_key,base_url = openai_api_base)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]
