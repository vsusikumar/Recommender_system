import streamlit as st

st.title("Recommender system")
st.sidebar.title("Upload your PDF here")
corpus_file = st.sidebar.file_uploader("Corpus", type="pdf")


if corpus_file is not None:


    option = st.sidebar.selectbox(
        'Select type of recommendation',
        ('Check content in your document','Grammar recommendation', 'Content recommendation'))
    if (option == 'Check content in your document'):
        st.header("Document viewer")


    elif (option == 'Grammar recommendation'):
        st.header("Set of Grammer Recommendation")





        st.header("Check sentiment for each sentence in slides")



    elif (option == 'Content recommendation'):
        st.header("Set of Content Recommendation")



