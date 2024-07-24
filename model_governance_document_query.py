import glob
import json
import os

import pandas as pd
import torch

from typing import Generator

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.language_models import LLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (load_prompt, SystemMessagePromptTemplate, HumanMessagePromptTemplate,
                                    ChatPromptTemplate)
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain


#
# Gets an instance of an LLM for all the llm processing.  It will first see if there are any materialized GGUF
# format files in the local directory.  If there is, it will use LlamaCPP to materialize the binary into an LLM
# instance.  If no GGUF file is present, then it will instead use the default openai chat gpt endpoint for the LLM
# model.
#
def get_llm_instance() -> LLM:
    llm_files = glob.glob('./*.gguf')
    if llm_files:
        if len(llm_files) > 1:
            raise FileNotFoundError("More than one GGUF file was present, so we can not know which one to utilize")
        llm = LlamaCpp(model_path=os.path.abspath(llm_files[0]),
                       model_kwargs={"max_length": 10000}, n_ctx=2048,
                       n_gpu_layers=80 if torch.cuda.is_available() else 0, verbose=True, top_p=1, temperature=0.2)
    else:
        llm = ChatOpenAI(temperature=0.2, max_tokens=2048)

    return llm


#
# The score function is the function called by modelop center engine on a given request.  It will receive one row of
# data from the data file and in the case of a batch process will be called repeatedly until the process is complete.
# In this case, it first instantiates either a local LLM instance, if a gguf file exists in the deployment, otherwise
# it will utilize the default openai model as a vendor endpoint.  It will then take the input and look for the question
# field, and will then populate a response based on the vectorized model governance documents, and return the results.
# To continue the conversation with the model, send back in the chat_history for context.
#

# modelop.score
def score(data: dict) -> Generator[dict, None, None]:
    llm = get_llm_instance()
    system_prompt = load_prompt("prompts/system_prompt.json")
    user_prompt = load_prompt("prompts/user_prompt.json")
    messages = [
        SystemMessagePromptTemplate(prompt=system_prompt),
        HumanMessagePromptTemplate(prompt=user_prompt)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(embedding_function=embedding_function, persist_directory="./chroma_db",
                       collection_name="governance_docs")
    chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(),
                                                  return_source_documents=True, verbose=True,
                                                  combine_docs_chain_kwargs={'prompt': qa_prompt})
    chat_history = []
    for history in data.get('chat_history', []):
        chat_history.append(tuple(history))
    llm_result: dict = chain.invoke({
        "question": data.get("question"),
        "chat_history": chat_history})
    result = {"question": llm_result.get("question", ""),
              "answer": llm_result.get("answer", ""),
              "chat_history": llm_result.get("chat_history", []),
              "source_documents": data.get('chat_history', [])}
    result["chat_history"].append([llm_result["question"], llm_result["answer"]])
    for document in llm_result.get("source_documents", []):
        result["source_documents"].append({
            "document": document.metadata.get("source", "unknown"),
            "start_index": document.metadata.get("start_index", 0),
            "page_content": document.page_content
        })

    yield result


#
# Main is used for testing the application.  We send two related requests into the model in chat format in order to make
# sure the chat history function is working correctly.
#
def main():
    result = next(score({"question": "What penalties occur if I do not govern my models properly"}))
    print(json.dumps(result, indent=2))
    result = next(score({"question": "What would that value be in USD?", "chat_history": result.get("chat_history")}))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
