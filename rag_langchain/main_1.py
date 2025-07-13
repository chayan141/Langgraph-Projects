import os
import json
import numpy as np
import requests
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
from dotenv import load_dotenv
load_dotenv()
import os
gemini = os.getenv('gemini_api_key')
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=gemini, temperature=0.5)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07", google_api_key=gemini)

vector = embedding_model.embed_query("hello, world!")
print(vector[:5])


#loading the document
with open('data/founder_story.txt','r', encoding='utf-8') as f:
    text = f.read()
    print(text)


# chunking the data
chunks = [text[i:i+500] for i in range(0, len(text),500)]
docmuents = [Document(page_content=chunk) for chunk in chunks]


#vector store
faiss_index = FAISS.from_texts(
    texts = [doc.page_content for doc in docmuents],
    embedding=embedding_model
)

#retriever
retriever = faiss_index.as_retriever()

# tools 


def calculator_tool(query):
    """evaluate the arithmetic expression"""
    try:
        return str(eval(query))
    except Exception as e :
        return f"error : {e}"
    
def summarizer_tool(text):
    """Summarize any text"""
    return model.invoke([
        {"role": "system", "content": "You summarize content."},
        {"role": "user", "content": f"Summarize:\n{text}"}
    ])
    
import wikipedia

def wikipedia_tool(query):
    """Fetch a summary from Wikipedia."""
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"Error: {e}"


def translate_tool(text_and_language):
    """
    Translate text to a target language.
    Example input: 'Hello World || French'
    """
    try:
        parts = text_and_language.split("||")
        text = parts[0].strip()
        target_language = parts[1].strip()
    except:
        return "Invalid input format. Use: Text || Language"

    prompt = [
        {"role": "system", "content": "You translate text."},
        {"role": "user", "content": f"Translate this into {target_language}:\n{text}"}
    ]
    return model.invoke(prompt)

def explain_code_tool(code):
    """Explain what this code does."""
    prompt = [
        {"role": "system", "content": "You are an expert programmer who explains code."},
        {"role": "user", "content": f"Explain what this code does:\n{code}"}
    ]
    return model.invoke(prompt)


#Tool List
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="MUST be used for ANY calculation request. Always use this tool to compute math expressions."

    ),
    Tool(
        name="Summarizer",
        func=summarizer_tool,
        description="Summarizes any text provided."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia_tool,
        description="Searches Wikipedia and returns a summary. Input should be the search term."
    ),
    Tool(
        name="Translator",
        func=translate_tool,
        description="Translates text into a target language. Input format: 'Text || Language'."
    ),
    Tool(
        name="CodeExplainer",
        func=explain_code_tool,
        description="Explains what a code snippet does."
    )
]

# memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#agent
agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

#chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    # Use RetrievalQA first
    retrieved_answer = qa_chain({"query": user_input})["result"]

    # Let the agent decide whether to use tools/memory
    final_response = agent.invoke(f"{user_input}\nRetrieved Info: {retrieved_answer}")

    print("\n[DEBUG] Memory so far:")
    for m in memory.chat_memory.messages:
        print(f"{m.type.upper()}: {m.content}")
    
    print(f"\nBot: {final_response['output']}\n")




