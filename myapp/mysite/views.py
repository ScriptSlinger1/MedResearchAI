from django.conf import settings
from django.http import HttpRequest
from django.shortcuts import render
from django.views.generic import TemplateView
from openai import OpenAI
import requests
import xml.etree.ElementTree as ET


# Create your views here.

class LandingPageView(TemplateView):
    template_name = 'mysite/landing.html'


def research_page_view(request: HttpRequest):
    response_text = None
    articles = []
    codes = []
    research_context = ""

    template_name = 'mysite/research_page.html'

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    if request.method == "POST":
        question = request.POST.get('med_question')

        # 1. Search PubMed
        codes = search_pubmed(question)

        # 2. Retrieve papers
        articles = fetch_pubmed_articles(codes)

        # 3. Build research context
        research_context = build_research_context(articles)

        # 4. Give the research to GPT
        prompt = f"""
        You are a medical research assistant.

        Answer the user's question using ONLY the scientific
        sources provided below.

        User question:
        {question}

        Scientific sources:
        {research_context}

        Instructions:
        - Base your answer on the provided sources.
        - Do not invent facts or citations.
        - If the sources do not provide enough evidence to answer
          the question, say so.
        - Clearly distinguish established findings from uncertain
          or conflicting evidence.
        - Do not provide a medical diagnosis or personalized
          medical advice.
        """

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        response_text = response.output_text

    return render(
        request,
        template_name,
        {
            'response': response_text,
            'articles': articles,
            'codes': codes,
            'context': research_context
        }
    )


def search_pubmed(question):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": question,
        "retmode": "json",
        "retmax": 13,
        "tool": "MedResearchAI",
        "email": "kimmax1503@gmail.com"
    }

    response = requests.get(url, params=params)

    data = response.json()

    return data["esearchresult"]["idlist"]


def fetch_pubmed_articles(codes):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(codes),
        "retmode": "xml",
        "tool": "MedResearchAI",
        "email": "your@email.com"
    }

    response = requests.get(url, params=params)

    root = ET.fromstring(response.text)

    articles = []

    for article in root.findall(".//PubmedArticle"):

        # PMID
        pmid = article.findtext(".//PMID")

        # Title
        title = article.findtext(".//ArticleTitle")

        # Abstract
        abstract_parts = article.findall(".//AbstractText")

        abstract = " ".join(
            part.text or ""
            for part in abstract_parts
        )

        # Authors
        authors = []

        for author in article.findall(".//AuthorList/Author"):

            last_name = author.findtext("LastName")
            fore_name = author.findtext("ForeName")

            if last_name and fore_name:
                authors.append(f"{fore_name} {last_name}")

            elif last_name:
                authors.append(last_name)

        # Journal
        journal = article.findtext(".//Journal/Title")

        # Dictionary
        article_data = {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "journal": journal,
            "abstract": abstract
        }

        articles.append(article_data)

    return articles


def build_research_context(articles):
    context = ""

    for article in articles:
        context += f"""
    Title: {article['title']}
    Authors: {", ".join(article['authors'])}
    Journal: {article['journal']}
    PMID: {article['pmid']}
    
    Abstract:
    {article['abstract']}
    
    -------------------------
    """

    return context
