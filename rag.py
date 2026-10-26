import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables import RunnableWithMessageHistory
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 优先读环境变量；没有时回退到原来写死的那把 key（建议之后只保留环境变量）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    return prompt

class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model,dashscope_api_key=DASHSCOPE_API_KEY),
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我的参考资料为主，简洁和专业的回答用户问题，参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        # qwen3.8-27b 只有走 DashScope 的 OpenAI 兼容模式端点才能调用。
        # ChatTongyi 走的是原生 /api/v1/services/aigc/text-generation/generation，
        # 且它没有 base_url 参数，传了也会被忽略，所以这里改用 ChatOpenAI。
        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            api_key=DASHSCOPE_API_KEY,
            base_url=config.chat_base_url,
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        # 获取最终执行链
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            """把结构化的 List[Document] 序列化成一段纯文本。"""
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def tmp2(value):
            # {input, context, history}
            new_value={}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(tmp2) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain