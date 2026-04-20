from groq import Groq
from dotenv import load_dotenv
import os, json

load_dotenv()

# ===============================
# All valid document categories
# ===============================
CATEGORIES = [
    "VMGO", "PEO", "PO", "Faculty", "Curriculum", "Instruction", "Students",
    "Research", "Extension", "Library", "Facilities", "Laboratories",
    "Administration", "Institutional Support", "Strategic Planning",
    "Special Orders", "DPCR", "IPCR", "Budget", "Activity Report",
    "Memorandum", "Minutes of Meeting", "Transmittal Letter", "Documentation",
    "Best Practice", "Audit", "Client Satisfactory", "Quality Objectives",
    "Risk Registers", "Trainings", "PES", "Faculty Advising",
    "Faculty Consultation", "Class Interventions", "Student Internship",
    "Approved Leave", "Daily Time Records (DTR)", "Faculty Fellowship Contracts",
    "Notarized Contracts", "Terms of Reference (TOR)", "Institutional Records",
    "Quality Assurance",
]


def call_llm(prompt: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    if not os.getenv("GROQ_API_KEY"):
        return {"error": "GROQ_API_KEY not found in environment"}

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You must respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        print("RAW:", raw)

        return json.loads(raw)

    except Exception as e:
        return {
            "error": str(e),
            "raw_output": response.choices[0].message.content if "response" in locals() else None
        }


def classify_document(text: str, svm_candidates: list[dict] | None = None) -> dict:
    """
    Send extracted document text + SVM candidate categories to the LLM
    for final classification and tagging.

    Parameters:
      text            – OCR-extracted document text
      svm_candidates  – ranked list from svm_classify(), e.g.
                        [{"category": "Research", "confidence": 0.42}, ...]

    Returns:
      {
        "primary_category":   str,
        "secondary_category": str,
        "tags":               list[str]
      }
    On failure returns the error dict from call_llm.
    """
    categories_list = "\n".join(f"- {c}" for c in CATEGORIES)

    # Build the SVM hint section
    svm_hint = ""
    if svm_candidates:
        ranked = "\n".join(
            f"  {i+1}. {c['category']} (confidence: {c['confidence']:.2%})"
            for i, c in enumerate(svm_candidates)
        )
        svm_hint = f"""
An SVM classifier has pre-analysed this document and ranked these as the most
likely categories (in order of confidence):
{ranked}

Use these SVM predictions as a strong guide, but you MAY override them if the
document text clearly belongs to a different category.
"""

    prompt = f"""You are a document classification assistant for a Quality Assurance Management System (QuAMS).

Below is the extracted text of a document from OCR:
\"\"\"
{text[:4000]}
\"\"\"
{svm_hint}
Your task:
1. Choose the PRIMARY category (the one you are MOST confident about).
2. Choose the SECONDARY category (the next best match, semi-confident).
3. Generate a list of short, relevant tags that describe the document content
   (e.g. "framework", "research design", "outreach", "quarterly report").
   Aim for 3-5 tags.

You MUST choose ONLY from this list of categories:
{categories_list}

Respond ONLY with a valid JSON object in this exact format:
{{
  "primary_category": "<category from list>",
  "secondary_category": "<category from list>",
  "tags": ["tag1", "tag2", "tag3"]
}}"""

    result = call_llm(prompt)
    return result

