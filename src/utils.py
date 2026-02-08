import re
import numpy as np


topic_names = {
    0: "SD/Memory Cards",
    1: "Product Functionality / Satisfaction",
    2: "Samsung Devices",
    3: "Micro SD Cards",
    4: "SanDisk Brand / Quality",
    5: "Memory / Storage",
    6: "Price & Quality",
    7: "Card Class / Speed (Class 10, UHS)",
    8: "Tablets / Android Tablets",
    9: "GoPro Cameras",
    10: "Works Fine / No Issues",
    11: "Phones / Smartphones",
    12: "Price / Good Deal",
    13: "Memory Size / GB",
    14: "Product / Recommendations",
    15: "Speed / Reliability",
    16: "Usage / Ease of Use",
    17: "Samsung Note Series",
    18: "Basic Usage / Simple Functions",
    19: "Performance / Does Job",
    20: "Read/Write Speed",
    21: "Media / Music & Videos",
    22: "Delivery / Time",
    23: "Samsung Galaxy S3",
    24: "Samsung Galaxy S4",
    25: "Problems / Issues",
    26: "Misc / Packaging & Labels",
    27: "Cards / Multiple Types",
    28: "Issues / Complaints",
    29: "Microsoft Surface / Tablets",
    30: "Adapters / Accessories",
    31: "Cameras / Video Recording",
    32: "Purchases / Bought Items",
    33: "Worked / Success Stories",
    34: "Feedback / User Opinions",
    35: "Storage Capacity / Space",
    36: "Buying / Recommendations",
    37: "Recommendations / Highly Recommended",
    38: "Likes / User Satisfaction",
    39: "Usage History / Past Experience",
    40: "Samsung Galaxy Tab",
    41: "Perfect / Fits Well",
    42: "Formatting / File System",
    43: "Purchases / Orders",
    44: "File Transfer / Speed",
    45: "Data Handling / Loss",
    46: "Disk / Brand Quality",
    47: "Work / Functionality",
    48: "Storage Space / Extra Room",
    49: "Perfectly / No Problems"
}


def predict_topic(text, pipeline, topic_names=topic_names):
    W = pipeline.transform([text])
    topic_id = int(W.argmax(axis=1)[0])
    score = float(W.max(axis=1)[0])
    return {
        "topic_id": topic_id,
        "topic_name": topic_names[topic_id],
        "confidence": score
    }
    
def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]

def _tokenize_words(sent):
    return re.findall(r"\b\w+\b", sent.lower())


def summarize_review(text, word2idf, top_k=1):
    text = str(text).strip()
    if not text:
        return ""
    sentences = _split_sentences(text)
    if len(sentences) <= top_k:
        return text
    scores = []
    for sent in sentences:
        words = _tokenize_words(sent)
        weights = [word2idf.get(w, 0) for w in words if w in word2idf]
        score = np.mean(weights) if weights else 0
        scores.append(score)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return " ".join(sentences[i] for i in sorted(top_indices))