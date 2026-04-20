import streamlit as st 
import requests

api_key = "6b610bda22b1425d96a9d79608780b45"

st.title("Yangiliklar Oynasi")

query = st.text_input("Search news", "AI")


if st.button("Yangiliklarni qidirish"):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "apiKey" : api_key
    }

    res = requests.get(url, params=params)
    data = res.json()

    for article in data["articles"][:25]:
        st.write("###", article["title"])
        st.write(article["source"]["name"])
        st.write(article["url"])