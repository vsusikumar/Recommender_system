
import streamlit as st


import PyPDF2

import textract
import nltk
import pandas as pd

import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

model_name = 'deep-learning-analytics/GrammarCorrector'

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)


text_to_list= []

pdf_dataframe=[]
ref_lst=[]

preprocessed=[]
grammer_list=[]
Classification_list=[]
Obj_del_txt=[]
sentiment_lst=[]





def read_pdf(corpus_file):
    pdfFileObj = corpus_file

    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    for x in range(pdfReader.numPages):
        if x != 8 and x != 4 and x!= 5:
            pageObj = pdfReader.getPage(x)
            text_x = pageObj.extractText()

            check1 = text_x.replace(" \n", " ")
            text_list = check1.split("\n")

            text_to_list.append(text_list)
            for i in text_list:
                if len(i) > 7:
                    count = x + 1
                    pdf_dataframe.append(['Slide ' + str(count), i])
        elif x==8:
            pageObj1 = pdfReader.getPage(x)
            text_x1 = pageObj1.extractText()
            print("Refrence text",text_x1)


            check2 = text_x1.replace(" \n", " ")
            check2 = check2.replace("\n ", " ")
            check2 = check2.replace("\n,", " ")
            #check2 = check2.replace(".\n", " ")
            check2 = check2.replace("\n.", " ")
            check2 = check2.replace("-\n", "- ")
            check2 = check2.replace("\n-", " -")
            check2 = check2.replace("&\n", " ")

            #check2 = check2.replace(".\n", " ")
            text_list_1= check2.split("\n")

            for i in text_list_1:
                if len(i) > 15 and i.find('@') == -1:
                    ref_lst.append(i)
            print('Text after split', ref_lst)



    df = pd.DataFrame(pdf_dataframe, columns=['Slide number', 'Content'])

    return df


def correct_grammar(input_text,num_return_sequences):
  batch = tokenizer([input_text],truncation=True,padding='max_length',max_length=64, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=200,num_beams=4, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text


def sentiment():
    for i in preprocessed:

        x=str(i[1])
        y=str(i[0])
        sent = sentiment_analysis(x)
        for i in sent:
            sentiment_lst.append([y,x, i['label']])

    df = pd.DataFrame(sentiment_lst, columns=['Slide','Content', 'Sentiment'])
    return df










def check_references():
    #print(ref_lst)
    sorted_lst=sorted(ref_lst)
    #print(sorted_lst)

    if sorted_lst == ref_lst:
        st.success("Reference are arranged alphabetical order")

    else :
        st.warning("Please double-check your references in your presentation and alphabetize them.")


def check_sentenece():

    for x in pdf_dataframe:
        #print("list", x[0])
        if x[1].find('@') == -1 and len(x[1]) > 20 and x[0] !='Slide 9':
           # print('Inside the loop ')
            if x[1].find('Gate 0') == -1:
                if (x[1] != 'Technology Innovation Management' and x[1] != 'Objective and deliverables'):
                    if x[1].endswith('.'):
                        x[1] = x[1]

                    else:
                        x[1] = x[1] + '.'
                    preprocessed.append(x)
                    #print("Preprocesse0",preprocessed)


def checkrecommendation():

    for i in preprocessed:

        x=str(i[1])
        y=str(i[0])

        rec_c=correct_grammar(x,1)
        #print("type++",type(rec_c))


        for j in rec_c :
          #print("type++ yyy",y)
          #print("outside if loop", i[1])
          if x != j :
              #print("inside if loop",y)
              grammer_list.append([y,x,j])
        #print("testing recommendation",correct_grammar(x[1],1))

    df = pd.DataFrame(grammer_list, columns=['Slide number','Sentence', 'Recommendation'])
    return df





def check_pages(corpus_file):
    pdfFileObj = corpus_file

    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    print("number of pages",pdfReader.numPages)
    if pdfReader.numPages > 9:
        st.warning("Please follow the Gate 0 you can have only 9 slides in your presentation")
    else:
        st.success("Your presentation has followed Gate 0 template")
    for x in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(x)
        text_extract = pageObj.extractText()
        print("Type of the text extracted",type(text_extract))








#streamlit UI
st.title("Recommender system")
st.sidebar.title("Upload your PDF here")
corpus_file = st.sidebar.file_uploader("Corpus", type="pdf")


if corpus_file is not None:

    Document=read_pdf(corpus_file)
    check_sentenece()
    option = st.sidebar.selectbox(
        'Select type of recommendation',
        ('Check content in your document','Grammar recommendation', 'Content recommendation'))
    if (option == 'Check content in your document'):
        st.header("Document viewer")
        st.dataframe(Document,width=900, height=300)

    elif (option == 'Grammar recommendation'):
        st.header("Set of Grammer Recommendation")
        check_data = checkrecommendation()


        st.dataframe(check_data, width=1000, height=300)

        st.header("Check sentiment for each sentence in slides")
        sentiment_data = sentiment()

        st.dataframe(sentiment_data, width=1000, height=300)
    elif (option == 'Content recommendation'):
        st.header("Set of Content Recommendation")
        check_references()
        check_pages(corpus_file)
        #text_classification()





    #text_Classification()






