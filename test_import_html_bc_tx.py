"""
Global example of parsing HTML to extract information.
here we use ChromaDB + Ollama (emb + LLM chat) + Langchain
- We have a report from polygonscan about a blockchain transaction 
   - multi wallet from / to 
   - multi tokens are distributed
- The task is to find out what happened in the transaction : 
    - to or from a given wallet address
    - which tokens were transfered

- DONE : 
    - What are the tokens exchanged with the wallet ?
    - Used different Re récursive HTML splitter to import an HTML file
    - Test, With One Example
- TODO :
    - Add from / to information, Smart Contract Transaction address
    - Add quantity of transaction
"""
import os
import time
from libs.utilities import getconfig
from transformers import GPT2Tokenizer
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
# from langchain_community.document_transformers import DocumentTransformer
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.wait import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from dotenv import load_dotenv, find_dotenv
# from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List
from libs.tools_ollama import launch_server_ollama

# definitions
MODE_UPDATE_DB = True  # True : Always re-construct vector store for RAG
config = "main"
# Embedder: Ollama model
embedmodel = getconfig(config)["embedmodel"]
# LLM Ollama model : actually, check if it exists or pull it and prepare it
mainmodel = launch_server_ollama(config=config)
pathdata = getconfig(config)["pathdata"]
relative_path_db = getconfig(config)["dbpath"]
collectionname = "tx_test"
url_test = "https://polygonscan.com/tx/0x99e3c197172b967eb4215249be50034a1696423a9ae805438ae217a501d86aa9"
file_path_test = "content/file_test_polygonscan.html"  # local download of remote  HTML file
address_test = "0x8da02d597a2616e9ec0c82b2b8366b00d69da29a"  # address of the wallet to scan
'''0x8da02D59...0d69da29A received 110.613389293981481481 Aavegotchi F... (FUD)
0x8da02D59...0d69da29A received 112.471063425925925925 Aavegotchi F... (FOMO)
0x8da02D59...0d69da29A received 25.951873206018518518 Aavegotchi A... (ALPHA)
0x8da02D59...0d69da29A received 10.775977939814814814 Aavegotchi K... (KEK)'''
# tokens to find for this test
dict_tokens_to_find = {
    "FOMO":  "0x44a6e0be76e1d9620a7f76588e4509fe4fa8e8c8",
    "FUD": "0x403e967b044d4be25170310157cb1a4bf10bdd0f",
    "KEK": "0x42e5e06ef5b90fe15f853f59299fc96259209c5c",
    "ALPHA": "0x6a3E7C3c6EF65Ee26975b12293cA1AAD7e1dAeD2",
}

# Define your desired data structure.


class TokenData(BaseModel):
    name: str = Field(description="name of the coin found")
    symbol: str = Field(description="symbol of the coin found")
    address: str = Field(description="address of the coin found")
    # You can add custom validation logic easily with Pydantic.

    @validator("address")
    def address_start_with_0x(cls, field):
        """Check address format"""
        if field[:2] != "0x":
            raise ValueError("Badly formed address!")
        return field


class ListTokenData(BaseModel):
    tokens: List[TokenData] = Field(description="list of coins found")


# Download with BeautyfulSoup
if not os.path.isdir(pathdata):
    os.mkdir(pathdata)

if not os.path.isfile(file_path_test):
    print("Loading url : ", url_test)
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    # get html content from chrome driver
    driver.get(url_test)
    # parse final content
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")
    # save html
    with open(file_path_test, "w", encoding="utf-8") as file:
        file.write(str(soup))
    # close browser
    driver.close()
    driver.quit()
else:
    # reload file
    print("Loading file : ", file_path_test)


# load text from HTML file on disk (with tags) : 2500 chars to have one tx
# https://python.langchain.com/v0.2/docs/how_to/code_splitter/
loader = TextLoader(file_path_test)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML, chunk_size=5000, chunk_overlap=0
)
all_splits_raw = text_splitter.split_documents(docs)

# Filter only chuncks with address test
# filter_transformer = FiltreDocumentTransformer(address_test)
# all_splits = filter_transformer.transform_documents(all_splits)
all_splits = [doc for doc in all_splits_raw if address_test in doc.page_content]
nb_char = len(all_splits[5].page_content)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode(all_splits[5].page_content)
nb_tokens = len(tokens)
print(f"Nb Characters per Context: {nb_char}")
print(f"Nb Tokens per Context: {nb_tokens}")

# load it into Chroma if not already done
# connect to Ollama embedding model
embedder = OllamaEmbeddings(
    model=embedmodel,
    embed_instruction="",
    query_instruction="Represent this sentence for searching relevant passages: ",
)

# Connect to VectorStore
# client = chromadb.HttpClient(host="localhost", port=8000) # TO SERVER (TEST 04 : 85s)
# TO LOCAL PERSISTENT DB (TEST 04 : 85s same result)
client = chromadb.PersistentClient(path=relative_path_db)

if any(collection.name == collectionname for collection in client.list_collections()):
    print("docs already in collection")
    mode_add = False
else:
    mode_add = True

# force update DB ?
if MODE_UPDATE_DB:
    if not mode_add:
        client.delete_collection(collectionname)
        mode_add = True

# langchain chroma connection
vectorstore = Chroma(
    client=client,
    collection_name=collectionname,
    embedding_function=embedder,
)

if mode_add:
    starttime = time.time()
    print("Adding docs to vector store")
    vectorstore.add_documents(all_splits)  # VERY LONG ?
    print("Done adding docs to vector store")
    print("--- %s seconds ---" % (time.time() - starttime))
# query it
# query = "Which tokens were transfered in this transaction?"
# query = "Which tokens are transfered from or to this address '0x8da02d597a2616e9ec0c82b2b8366b00d69da29a'?"
# query = "Find all tokens transfered to this address '0x8da02D59...0d69da29A'. Outputs only the name and address of these tokens and nothing else."
# query = "Find tokens, their names and adresses, that were exchanged with this wallet '0x8da02D59...0d69da29A' without the receiver or destination wallet addresses."
# query = "Find tokens, their names and adresses, that were exchanged with this wallet '0x8da02d597a2616e9ec0c82b2b8366b00d69da29a' without the receiver or destination wallet addresses."
query = f"Find coins, their symbols and adresses, that were exchanged with this wallet '{address_test}' without the receiver or destination wallet addresses."

search_kwargs = {
    "k": 23,
    "where_document": {"$contains": address_test[:10]},
}
retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)  # only 10 docs
# LLM model
llm = Ollama(model=mainmodel, num_ctx=4092)


# TEST 0 :  try to extract FUD, FOMO, KEK, ALPHA from the context
#  but without specific output format
print('\nTEST 0 : Get only token names with RAG (VectorDB + Chat LLM) : \n')
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the context provided:
    <context>
    {context}
    </context>
    Question: {input}
    """
)
# Stuff all relevant docs (all docs in this case)
document_chain = create_stuff_documents_chain(llm, prompt)
# Add to the retriever to have the complete chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# REPONSE TEST :
results = retrieval_chain.invoke({"input": query})
print("\nTEST 0 : CHECK CONTEXT :\n")
# check context
n_context_example = None
for k, doc in enumerate(results["context"]):
    if doc.page_content.find("FUD") != -1:
        print("FOUND FUD in context   : ", k)
        n_context_example = k
    if doc.page_content.find("FOMO") != -1:
        print("FOUND FOMO in context  : ", k)
    if doc.page_content.find("KEK") != -1:
        print("FOUND KEK in context   : ", k)
    if doc.page_content.find("ALPHA") != -1:
        print("FOUND ALPHA in context : ", k)
if n_context_example is None:
    n_context_example = k
print("\nTEST 0 : RESULTS :\n")
print(results["answer"])
print("\nTEST 0 DONE.\n")

# TEST 1 : With output format + 1 example and the last 2 context found : json
print("\nTEST 1 : With output format on 2 context found and an example : \n")
# prepare an example :
results_old = results
context_ref = results_old["context"][n_context_example].page_content

for token, address_token in dict_tokens_to_find.items():
    if context_ref.find(token) != -1:
        print("EXAMPLE : FOUND TOKEN : ", token)
        if context_ref.find(address_token) != -1:
            print("EXAMPLE : FOUND ADDRESS : ", address_token)
            example = f"``` {context_ref} ```"
            output_example = """
            ```json
            {
            "name": [""" + '"Aavegotchi ' + token + '"' + """],
            "symbol": [""" + '"' + token + '"' + """],
            "address": [""" + '"' + address_token + '"' + """]
            }
            ```
            """
            break

# try to extract for all tx found in all context
# Set up a parser + inject instructions into the prompt template.
# prompt without example
query_2 = f"Find all coins name, symbol and address that were exchanged with this wallet '{address_test}' ?"
parser = PydanticOutputParser(pydantic_object=ListTokenData)
prompt_2 = PromptTemplate(
    template="""Answer the following question only based on the context (several pieces of a HTML file) and instructions provided.
        <context>
        {context}
        </context>
        <example>
        This example of html file part:
        {example}
        gives the output: 
        {output_example}
        </example>
        <instructions>
        {format_instructions}
        </instructions>
        <question>
         {query}
         Don't take wallet address of sender (From) or of receiver (To), but only token address.
         A symbol of a coin is a 3 to 5 characters string and use exclusively capital letters of the Latin alphabet (A-Z).
         A name of a coin is the long format string of the symbol
         An address of a coin is a 42 characters string starting with 0x
         A coin have only one address, so output only one address per coin.
         As answer, for each coin found, can you give his name, symbol and address.
         Use only this context to answer.
         Do not explain how you have done.
        </question>
        """,
    input_variables=["query", "example", "output_example", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
dict_param_prompt_2 = {
    "context": results_old["context"][-2].page_content + results_old["context"][-1].page_content,
    "example": example,
    "output_example": output_example,
    "query": query_2,
}

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt_2 | llm
output = prompt_and_model.invoke(dict_param_prompt_2)
print("\noutput to parser:\n", output)
res = parser.invoke(output)
print("\nAnswer parsed : \n")
print(res)
print("\nTEST 1 END")

# TEST 2 : With output format on all context found : json
print("\nTEST 2 : With output format on all docs and without example : \n")
starttime = time.time()

# declaration
dict_token_found = {}
list_tokens_found = []
# loop over contexts
for k, doc in enumerate(all_splits):
    print('\nContext n# ', k)
    # find token symbol and address
    dict_param_prompt_2["context"] = doc.page_content
    output = prompt_and_model.invoke(dict_param_prompt_2)
    print("1-Output to parser:\n", output)
    if k == 13:
        print(doc.page_content)
    try:
        res = parser.invoke(output)
        print("2-Answer parsed:\n")
        print(res)
        for token in res.tokens:
            if token.symbol not in dict_token_found:
                dict_token_found[token.symbol] = token.address
            if token not in list_tokens_found:
                list_tokens_found.append(token)
    except:
        print("2-NO MORE token found ?")
print("--- %s seconds ---" % (time.time() - starttime))
print(list_tokens_found)
ok = 0
for symbol, expected_value in dict_tokens_to_find.items():
    if symbol in dict_token_found:
        if dict_token_found[symbol] == expected_value.lower():
            print(symbol, " : ", dict_token_found[symbol], " OK!")
            ok += 1
if ok == len(dict_tokens_to_find):
    print("TEST 2 OK")
else:
    print("TEST 2 NOK!")

print("\nTEST 2 END")
