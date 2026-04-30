import re
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# =========================
# PDF PROCESSING
# =========================

def process_pdf(file_path, embeddings):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    full_text = "\n\n".join([doc.page_content for doc in documents])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore, full_text


# =========================
# INTENT DETECTION
# =========================

def detect_intent(question):
    q = question.lower()

    if "phone" in q or "contact number" in q:
        return "phone"

    if "email" in q:
        return "email"

    if "linkedin" in q:
        return "linkedin"

    if "portfolio" in q:
        return "portfolio"

    if "certification" in q:
        return "certifications"

    if "extra" in q or "activity" in q:
        return "activities"

    if "language" in q:
        return "languages"

    if "skill" in q:
        return "skills"

    if "experience" in q:
        return "experience"

    if "education" in q:
        return "education"

    if "project" in q:
        return "projects"

    if "job" in q or "role" in q:
        return "job_recommendation"

    if "summary" in q or "about" in q:
        return "summary"

    return "general"


# =========================
# FILTER DOCS
# =========================

def filter_docs(intent, docs):
    filtered = []

    for d in docs:
        text = d.page_content.lower()

        if intent == "skills":
            if any(k in text for k in ["skills", "tools", "technologies"]):
                filtered.append(d)

        elif intent == "experience":
            if any(k in text for k in ["experience", "intern", "developer", "company"]):
                filtered.append(d)

        elif intent == "education":
            if any(k in text for k in ["education", "university", "degree"]):
                filtered.append(d)

        elif intent == "projects":
            if "project" in text:
                filtered.append(d)

        elif intent == "certifications":
            if "certification" in text:
                filtered.append(d)

        elif intent == "languages":
            if "language" in text:
                filtered.append(d)

        elif intent == "activities":
            if any(k in text for k in ["activity", "club", "volunteer"]):
                filtered.append(d)

    return filtered if filtered else docs


# =========================
# CLEAN CONTEXT
# =========================

def clean_documents(docs):
    seen = set()
    cleaned = []

    for d in docs:
        text = d.page_content.strip()

        if text not in seen:
            seen.add(text)
            cleaned.append(text)

    return "\n\n".join(cleaned)

def remove_duplicates(text):
    lines = text.split("\n")
    seen = set()
    result = []

    for line in lines:
        l = line.strip()
        if l and l not in seen:
            seen.add(l)
            result.append(l)

    return "\n".join(result)


# =========================
# REGEX EXTRACTORS (IMPORTANT)
# =========================

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else "Email not found"


def extract_phone(text):
    match = re.search(r'(\+?\d[\d\s\-]{8,})', text)
    return match.group(0) if match else "Phone number not found"


def extract_linkedin(text):
    match = re.search(r'(https?://(www\.)?linkedin\.com/[^\s]+)', text)
    return match.group(0) if match else "LinkedIn not found"


def extract_portfolio(text):
    match = re.search(r'(https?://[^\s]+)', text)
    return match.group(0) if match else "Portfolio link not found"


# =========================
# EXPERIENCE CALCULATION
# =========================

def estimate_experience(text):
    years = re.findall(r'(\d{4})', text)

    if len(years) >= 2:
        years = [int(y) for y in years]
        return f"Approximate experience: {max(years) - min(years)} years"

    return "Experience duration not clearly found"


# =========================
# PROMPT BUILDER
# =========================

def build_prompt(intent, context, question):

    if intent == "skills":
        return f"""
Extract only skills.

Format:
- Bullet points

Context:
{context}
"""

    elif intent == "experience":
        return f"""
Extract only work experience.

Format:
Role – Company
- Key responsibility

Context:
{context}
"""

    elif intent == "education":
        return f"""
Extract education details.

If not found say "Education not found".

Context:
{context}
"""

    elif intent == "projects":
        return f"""
Extract projects.

Format:
Project:
- Description
- Tools

Context:
{context}
"""

    elif intent == "certifications":
        return f"""
Extract certifications only.

Context:
{context}
"""

    elif intent == "languages":
        return f"""
Extract languages known.

Context:
{context}
"""

    elif intent == "activities":
        return f"""
Extract extracurricular activities.

Context:
{context}
"""

    elif intent == "summary":
        return f"""
Create a short professional summary.

Context:
{context}
"""

    elif intent == "job_recommendation":
        return f"""
Based on skills and experience, suggest suitable job roles.

Do not guess beyond context.

Context:
{context}
"""

    else:
        return f"""
Answer based only on context.

Context:
{context}

Question:
{question}
"""


# =========================
# MAIN FUNCTION
# =========================

def generate_answer(question, vectorstore, full_text, llm):

    intent = detect_intent(question)

    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(question, k=3)
    local_context = clean_documents(filter_docs(intent, docs))

    # Global context (important)
    global_context = full_text[:3000]

    prompt = f"""
You are a professional AI career assistant analyzing a resume.

You have TWO sources:

1. FULL RESUME:
{global_context}

2. RELEVANT SECTIONS:
{local_context}

Instructions:
- Understand the FULL resume before answering
- Use RELEVANT SECTIONS for accuracy
- Do NOT repeat raw text
- Do NOT guess or invent details
- Always structure answers clearly
- Be concise but complete

Response Rules:
- Use bullet points when listing
- Group information logically
- Remove duplication
- Make answers readable and professional

Special Cases:
- Skills → group into categories
- Experience → structured roles with impact
- Education → exact format
- Job roles → realistic suggestions based on profile

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    # remove repeated lines
    answer = remove_duplicates(answer)

    # smart fallback
    if len(answer) < 20:
        answer = "The information is not clearly available in the document."

    return answer, docs