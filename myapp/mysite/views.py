from django.conf import settings
from django.http import HttpRequest
from django.shortcuts import render
from django.views.generic import TemplateView
from openai import OpenAI
import requests
import xml.etree.ElementTree as ET
import json


# Create your views here.

class LandingPageView(TemplateView):
    template_name = 'mysite/landing.html'


def research_page_view(request: HttpRequest):
    response_text = None
    articles = []
    codes = []
    research_context = ""

    analysis = None
    pubmed_query = None

    template_name = 'mysite/research_page.html'

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    if request.method == "POST":
        question = request.POST.get('med_question')

        # 1. Analyze the user's question
        analysis_text = analyze_question(question)
        analysis = json.loads(analysis_text)

        # 2. Build a PubMed query from the analysis
        pubmed_query = build_pubmed_query(analysis)

        # 3. Search PubMed
        codes = search_pubmed(pubmed_query)

        # 4. Retrieve the papers
        articles = fetch_pubmed_articles(codes)
        articles = rank_articles(articles)

        # 5. Build the research context
        research_context = build_research_context(articles)

        # 6. Give the retrieved evidence to GPT
        prompt = f"""
        You are MedResearch AI, an AI-powered medical literature research assistant.

        Your task is to answer the user's research question using ONLY the scientific
        sources provided below.

        User question:
        {question}

        Scientific sources:
        {research_context}

        CORE RULE:
        Your conclusion must match the strength of the retrieved evidence.
        Be direct and useful, but never make a stronger claim than the evidence supports.

        INSTRUCTIONS:

        1. ANSWER THE ACTUAL QUESTION
        - Directly address what the user asked.
        - Do not answer a broader or different question.
        - Do not avoid reaching a conclusion when the evidence supports one.

        2. EVIDENCE STRENGTH
        Prioritize evidence in approximately this order:
        - Systematic reviews and meta-analyses
        - Randomized controlled trials
        - Prospective or large observational studies
        - Other observational studies
        - Mechanistic, laboratory, animal, or in-vitro studies

        When higher-quality evidence conflicts with lower-quality evidence,
        give greater weight to the higher-quality evidence.

        3. DO NOT OVERCLAIM
        - The opening conclusion MUST NOT be stronger than the evidence.
        - Do not treat mechanistic evidence as proof of clinical benefit.
        - Do not treat an improvement in a surrogate outcome as proof of
          improved clinical outcomes.
        - For example, an improvement in blood pressure, biomarker levels,
          or ventricular mass does NOT by itself prove fewer heart attacks,
          strokes, hospitalizations, or deaths.
        - Distinguish clearly between surrogate outcomes and clinical outcomes.
        - Do not infer causation from observational associations.
        - Do not generalize findings beyond the population studied.

        4. HANDLE UNCERTAINTY CORRECTLY
        Use one of these opening formats:

        - "Yes —" when direct evidence supports the claim.
        - "No —" when direct evidence indicates the claim is not supported.
        - "The evidence suggests..." when evidence is suggestive but not definitive.
        - "The evidence is insufficient..." when there is not enough relevant evidence.

        Do NOT use "Probably yes" or "Probably no" unless the retrieved evidence
        directly addresses the question and provides a reasonably consistent answer.

        If evidence supports an intermediate conclusion, state that conclusion
        precisely rather than forcing a yes/no answer.

        5. CONFLICTING EVIDENCE
        - If studies disagree, explicitly say so.
        - Explain which evidence is stronger and why.
        - Do not hide contradictory findings.
        - Do not present one positive study as if it represents the entire literature.

        6. SOURCE QUALITY
        - Pay attention to study design, population, sample size, and outcome measured.
        - Prefer evidence that directly answers the user's question.
        - Do not give equal weight to a protocol, mechanistic study, observational
          study, and randomized clinical trial.
        - A study that only investigates a related outcome should be described
          as indirect evidence.

        7. CITATIONS
        - Only cite studies that are actually present in the provided sources.
        - Use the PMID when referring to a specific study.
        - Never invent a PMID, study, statistic, result, or citation.
        - If a source is a protocol, review, observational study, or other design,
          do not describe it as a randomized clinical trial.
        - If a study is retracted, explicitly identify it as retracted and do not
          use its findings as supporting evidence.

        8. SAFETY
        - You are a medical literature research assistant, not a doctor.
        - Do not diagnose the user.
        - Do not provide personalized medical advice.
        - Do not recommend that a specific person start, stop, or change treatment.
        - Keep the answer focused on what the scientific literature shows.
        
        9. AVOID REPETITION
        - Do not repeat the same limitation or conclusion multiple times.
        - Each section should add new information.
        - Keep Evidence focused on findings, Limitations focused on uncertainty,
          and Bottom line focused on the final answer.
        - Prefer concise, information-dense explanations over lengthy repetition.

        OUTPUT FORMAT:

            Answer:
            [1-2 sentences directly answering the user's question.]
            
            Evidence:
            [3-5 of the most relevant findings, prioritized by evidence quality.
            For each important study, briefly state the study design, population,
            main finding, and PMID.]
            
            Limitations:
            [Only the most important limitations, conflicting evidence,
            or gaps that affect the conclusion.]
            
            Bottom line:
            [One concise sentence answering the original question.]

        IMPORTANT:
        The answer must distinguish between:
        - evidence that a treatment changes a biological or surrogate marker, and
        - evidence that it improves meaningful clinical outcomes such as disease
          progression, cardiovascular events, hospitalization, or mortality.

        If the retrieved literature does not actually establish the outcome asked
        about, say that clearly rather than extrapolating from related findings.
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
            'context': research_context,

            # Debug information
            'analysis': analysis,
            'pubmed_query': pubmed_query,
        }
    )

def search_pubmed(question):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": question,
        "retmode": "json",
        "retmax": 20,
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
        "email": "kimmax1503@gmail.com"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    root = ET.fromstring(response.text)

    articles = []

    for article in root.findall(".//PubmedArticle"):

        pmid = article.findtext(".//PMID")
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

        # Publication date
        year = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//PubDate/MedlineDate")
        )

        # Publication types
        publication_types = []

        for pub_type in article.findall(".//PublicationType"):
            if pub_type.text:
                publication_types.append(pub_type.text)

        article_data = {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "publication_types": publication_types,
            "abstract": abstract
        }

        articles.append(article_data)

    return articles

def rank_articles(articles):
    priority = {
        "Meta-Analysis": 6,
        "Systematic Review": 5,
        "Randomized Controlled Trial": 5,
        "Clinical Trial": 4,
        "Cohort Studies": 3,
        "Observational Study": 3,
        "Case-Control Studies": 3,
        "Review": 2,
        "Case Reports": 1,
        "Editorial": 0,
        "Letter": 0,
    }

    for article in articles:

        score = 0

        for publication_type in article["publication_types"]:
            score += priority.get(publication_type, 0)

        article["evidence_score"] = score

    return sorted(
        articles,
        key=lambda article: article["evidence_score"],
        reverse=True
    )


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


def analyze_question(question):
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    prompt = f"""
    You are a medical literature search specialist.
    
    Analyze the following research question and convert it into search concepts
    that can be used to retrieve relevant scientific literature from PubMed.
    
    Question:
    {question}
    
    Return ONLY valid JSON in exactly this structure:
    
    {{
        "population_terms": [
            "term1",
            "term2",
            "term3"
        ],
        "intervention_terms": [
            "term1",
            "term2",
            "term3"
        ],
        "outcome_terms": [
            "term1",
            "term2",
            "term3"
        ],
        "intent": "treatment"
    }}
    
    Rules:
    
    1. POPULATION TERMS
    - Identify WHO is being studied.
    - Include useful synonyms, alternative descriptions, and common
      scientific terminology.
    - If the question has no specific population, return an empty list.
    - Do not invent a specific population that is not present in the question.
    
    2. INTERVENTION / EXPOSURE TERMS
    - Identify the treatment, drug, exposure, diagnostic method, or
      primary subject being investigated.
    - Include common synonyms, generic names, scientific terminology,
      and alternative spellings where useful.
    - Include approximately 3-6 terms.
    
    3. OUTCOME TERMS
    - Identify the disease, condition, effect, symptom, or outcome being studied.
    - Include medical terminology, synonyms, and common alternative names.
    - Include approximately 3-6 terms.
    
    4. SEARCH TERMS
    - Terms should be useful for PubMed retrieval, not explanations.
    - Prefer established medical/scientific terminology.
    - Do not copy the entire question into a field.
    - Avoid overly broad terms such as "health", "disease", "patients",
      or "treatment" unless they are genuinely necessary.
    - Do not include duplicate terms.
    - Keep terms concise.
    
    5. SEARCH STRATEGY
    - The terms within each concept will later be connected with OR.
    - Different concepts will later be connected with AND.
    - Therefore, make each group contain genuine alternatives for the
      same concept rather than unrelated concepts.
    
    6. INTENT
    intent must be exactly one of:
    treatment
    diagnosis
    causes
    symptoms
    prevention
    prognosis
    risk_factors
    general
    
    Example:
    
    Question:
    Does smoking increase the risk of lung cancer?
    
    Output:
    
    {{
        "population_terms": [],
        "intervention_terms": [
            "smoking",
            "cigarette smoking",
            "tobacco smoking",
            "tobacco use",
            "cigarette use"
        ],
        "outcome_terms": [
            "lung cancer",
            "lung neoplasms",
            "lung carcinoma",
            "pulmonary cancer",
            "bronchogenic carcinoma"
        ],
        "intent": "risk_factors"
    }}
    
    Another example:
    
    Question:
    Does metformin reduce cardiovascular risk in non-diabetic patients?
    
    Output:
    
    {{
        "population_terms": [
            "non-diabetic patients",
            "non-diabetic adults",
            "patients without diabetes"
        ],
        "intervention_terms": [
            "metformin"
        ],
        "outcome_terms": [
            "cardiovascular risk",
            "cardiovascular disease",
            "cardiovascular events",
            "myocardial infarction",
            "stroke",
            "cardiovascular mortality"
        ],
        "intent": "risk_factors"
    }}
    """

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text

def build_pubmed_query(analysis):

    population_terms = analysis.get("population_terms", [])
    intervention_terms = analysis.get("intervention_terms", [])
    outcome_terms = analysis.get("outcome_terms", [])

    def build_group(terms):
        return " OR ".join(
            f'"{term}"[Title/Abstract]'
            for term in terms
        )

    groups = []

    if intervention_terms:
        intervention_query = build_group(intervention_terms)
        groups.append(f"({intervention_query})")

    if population_terms:
        population_query = build_group(population_terms)
        groups.append(f"({population_query})")

    if outcome_terms:
        outcome_query = build_group(outcome_terms)
        groups.append(f"({outcome_query})")

    return " AND ".join(groups)
