import json, os
from typing import Sequence
from langchain_core.messages import message_to_dict, BaseMessage, messages_from_dict
from langchain_core.chat_history import BaseChatMessageHistory

# 实现通过会话id获取InMemoryChatMessageHistory类对象
def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")

# message_to_dict: 单个消息对象(BaseMessage类实例) -> 字典
# message_from_dict: [字典，字典....] -> [消息，消息...]
# AIMessage, HumanMessage, SystemMessage 都是 BaseMessage的子类
class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id                # 会话id

        # 不同会话id的存储文件，所在的文件夹路径
        self.storage_path = storage_path            # 完整的文件路径

        # 完整的路径
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保文件所在的父目录存在，否则后续写入文件时会因路径不存在而报错。
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """批量写入。注意基类约定：这个方法是 add_messages（复数），
        一次收到 [HumanMessage, AIMessage] 这样的消息列表；
        如果只写 add_message（单数），基类会逐条调用，参数变成单个消息。"""
        if not messages:
            return

        all_messages = list(self.messages)      # 已有的消息列表
        all_messages.extend(messages)           # 新的和已有的融合成一个 list

        # 将数据同步写入本地文件
        # 类对象写入文件 -> 一堆二进制
        # 为了方便，可以将BaseMessage消息转为字典（借助 json模块以json字符串写入文件）
        new_messages = [message_to_dict(m) for m in all_messages]

        # 将数据同步写入到本地文件中
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f, ensure_ascii=False, indent=2)

    @property   # @property修饰器将message方法变为成员属性
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)