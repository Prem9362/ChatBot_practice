import streamlit as st
from rag_pipeline import get_answer , load_pdf

st.title('RAG Chatbot')

filename='Android-Lollipop-Quick-Start-Guide.pdf'  # file upload
collection=load_pdf(filename)   # vector db is ready

if 'messages' not in st.session_state:
    st.session_state['messages']=[]
    st.session_state['history']=[]                                    #1
messages=st.session_state['messages']
summary=st.session_state['history']
question =st.chat_input()

if len (summary)>5:                                                     #2
    summary.pop(0)

if question:
    messages.append({'role':'user','content': question})
    summary.append({'role':'user','content': question})                 #5
    answer,context = get_answer(question,collection,summary)          #3                          # rag ,llm
    if answer:
        messages.append({'role':'assistant','content': answer})
        summary.append({'role':'assistant','content': answer})      #6
        with st.expander('sources'):
            st.write(context)

for msg in messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])