import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.summarization import summarize


def main():
    st.title("NLP with Streamlit")
    st.subheader("Discover your text with Natural Language Processing")

    st.write('This application currently offers just the Word cloud and text summarization,\
			 which is just the tip of the iceberg when it comes to NLP. Will try adding some more functionality later!')

    user_input = st.text_area("Enter the text here")

    if st.button('Generate Word Cloud'):

        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.write('Word Cloud:')

        lines = user_input.splitlines()

        text = ''

        for each in lines:
            text += ' '.join(each.split(' '))

        # Create and generate a word cloud image:
        wc = WordCloud(background_color="white").generate(text)

        # Display the generated image:
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")

        st.pyplot()

    if st.button('Summarize the text '):
        short_summary = summarize(user_input)
        st.write('Short Summary of the text:')
        st.write(short_summary)


if __name__ == '__main__':
    main()
