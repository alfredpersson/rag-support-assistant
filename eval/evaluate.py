"""
Evaluate the RAG pipeline on 10 Wix-domain Q&A pairs.
Scoring: keyword overlap — % of expected keywords found in the generated answer.
"""

import json
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import run

QA_PAIRS = [
    {
        "question": "How do I connect a custom domain to my Wix site?",
        "keywords": ["domain", "connect", "DNS", "nameservers", "settings"],
    },
    {
        "question": "How can I add a blog to my Wix website?",
        "keywords": ["blog", "add", "Wix Blog", "posts", "pages"],
    },
    {
        "question": "How do I accept online payments on Wix?",
        "keywords": ["payment", "Wix Payments", "checkout", "store", "accept"],
    },
    {
        "question": "How do I change the fonts on my Wix site?",
        "keywords": ["font", "text", "design", "style", "editor"],
    },
    {
        "question": "What is Wix SEO and how do I improve my site's ranking?",
        "keywords": ["SEO", "search", "Google", "keywords", "meta"],
    },
    {
        "question": "How do I add members and a login page to my Wix site?",
        "keywords": ["members", "login", "signup", "account", "area"],
    },
    {
        "question": "How can I create a contact form on Wix?",
        "keywords": ["contact", "form", "Wix Forms", "submit", "email"],
    },
    {
        "question": "How do I set up Wix eCommerce and add products?",
        "keywords": ["store", "product", "eCommerce", "inventory", "add"],
    },
    {
        "question": "How do I restore a previous version of my Wix site?",
        "keywords": ["restore", "version", "history", "undo", "backup"],
    },
    {
        "question": "How do I make my Wix site mobile-friendly?",
        "keywords": ["mobile", "responsive", "editor", "view", "layout"],
    },
]


def score(answer: str, keywords: list[str]) -> float:
    answer_lower = answer.lower()
    matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return matched / len(keywords)


def main():
    results = []
    scores = []

    header = f"{'#':<4} {'Score':>6}  Question"
    print(header)
    print("-" * 70)

    for i, qa in enumerate(QA_PAIRS, start=1):
        question = qa["question"]
        keywords = qa["keywords"]

        try:
            result = run(question)
            answer = result["answer"]
            sources = result["sources"]
            error = None
        except Exception as e:
            answer = ""
            sources = []
            error = str(e)

        s = score(answer, keywords)
        scores.append(s)

        short_q = question[:55] + "..." if len(question) > 55 else question
        print(f"{i:<4} {s:>5.0%}   {short_q}")
        if error:
            print(f"     ERROR: {error}")

        results.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "keywords": keywords,
            "score": round(s, 4),
            "error": error,
        })

    mean = sum(scores) / len(scores)
    print("-" * 70)
    print(f"{'Mean score:':>12} {mean:.0%}")

    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump({"mean_score": round(mean, 4), "results": results}, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
