import sys
from unstract.llmwhisperer.client import LLMWhispererClient, LLMWhispererClientException
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

default_file = "./assets/docs/HSBC.pdf"
default_page = "16"

def process_HSBC(use_llamaparse=False, file = default_file, table_page = default_page):
    extracted_text = extract_text_from_pdf(default_file, pages_list=table_page)
    print(extracted_text)
    return extracted_text

def error_exit(error_message):
    print(error_message)
    sys.exit(1)

def extract_text_from_pdf(file_path, pages_list=None): #use llm whisperer to extract
    llmw = LLMWhispererClient()
    try:
        result = llmw.whisper(file_path=file_path, pages_to_extract=pages_list)
        extracted_text = result["extracted_text"]
        return extracted_text
    except LLMWhispererClientException as e:
        error_exit(e)

def extract_text_from_pdf_with_llamaparse(file_path, pages_list=None):
    # set up parser
    parser = LlamaParse(
        result_type="markdown",  # "markdown" and "text" are available
        target_pages=pages_list
    )

    # use SimpleDirectoryReader to parse our file
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
    extracted_text = ''
    for doc in documents:
        extracted_text += doc.text

    return extracted_text

def compile_template_and_get_llm_response(preamble, extracted_text, pydantic_object):
    postamble = "Do not include any explanation in the reply. Only include the extracted information in the reply."
    system_template = "{preamble}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{format_instructions}\n\n{extracted_text}\n\n{postamble}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    parser = PydanticOutputParser(pydantic_object=pydantic_object)


    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    request = PromptTemplate.from_template(preamble=preamble,
                                        format_instructions=parser.get_format_instructions(),
                                        extracted_text=extracted_text,
                                        postamble=postamble).to_messages()
    chat = ChatOpenAI()
    response = chat(request, temperature=0.0)
    print(f"Response from LLM:\n{response.content}")
    return response.content

