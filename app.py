import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)

# Streamlit APP
st.set_page_config(
    page_title="Summarize Text From YT or Website",
    page_icon="🤖"
)

st.title("Summarize Text From YT or Website")
st.subheader("Summarize URLs or YouTube Videos")


# Sidebar
with st.sidebar:
    groq_api_key = st.text_input(
        "Enter your Groq API Key",
        type="password"
    )

url = st.text_input(
    "URL",
    label_visibility="collapsed"
)


# Prompt
prompt_template = """
Provide a concise summary of the following content in 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# Button
if st.button("Summarize the content from YT or Website"):

    # Validate inputs first
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide the Groq API key and URL.")

    elif not validators.url(url):
        st.error("Please provide a valid URL.")

    else:
        try:
            with st.spinner("Loading....."):

                # Create LLM only after API key is available
                llm = ChatGroq(
                    api_key=groq_api_key,
                    model="openai/gpt-oss-20b"
                )

                # Load data
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(
                        url,
                        add_video_info=True
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )

                data = loader.load()

                # Combine documents
                text = "\n\n".join(
                    doc.page_content
                    for doc in data
                )

                # Modern LCEL chain
                chain = prompt | llm

                response = chain.invoke({
                    "text": text
                })

                st.success(response.content)

        except Exception as e:
            st.exception(e)