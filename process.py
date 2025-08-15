import sys
from datetime import datetime
from dotenv import load_dotenv
from unstract.llmwhisperer.client import LLMWhispererClient, LLMWhispererClientException
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import util.extract as ex

default_file = "./assets/docs/HSBC.pdf"
default_page = "1"



def process_HSBC(use_llamaparse=False, file = default_file, table_page = default_page):
    if use_llamaparse:
        extracted_text = ex.extract_text_from_pdf_with_llamaparse(default_file, pages_list=table_page)
    else:
        extracted_text = ex.extract_text_from_pdf(default_file, pages_list=table_page)
    print(extracted_text)
    return extracted_text
    #response = extract_receipt_details_from_text(extracted_text)
    #print(response)

