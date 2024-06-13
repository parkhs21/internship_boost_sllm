from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOllama(base_url="http://192.168.115.38:11510", model="gemma:2b")

llm.invoke()