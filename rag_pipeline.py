from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model = 'openai/gpt-oss-120b')

def load_pdf(filename):
    reader = PdfReader('Android-Lollipop-Quick-Start-Guide.pdf')
    text=''

    for page in reader.pages :
        text = text + page.extract_text()

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = splitter.split_text(text)

    client=chromadb.Client()
    collection = client.get_or_create_collection('Android_pdf')
    collection.add(documents=chunks
               ,ids=['c'+ str(i) for i in range(len(chunks))])

    return collection

def get_answer(question,collection,summary):                                        #4
    result = collection.query(query_texts=[question],n_results=2)
    context = result['documents'][0]
    prompt = '''
    I will provide you question and the context which may have answer of given question.

    You have to generate answer in 2 lines with respect to given context.
    If you dont find answer in given context, just reply "Answer Not found in given context".

    Use this {2} as summary of our old conversation.

    Question: {0}
    Context: {1}
    '''.format(question, context, summary)
    result = llm.invoke(prompt)
    return (result.content,context)