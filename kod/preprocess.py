import re
import ast
from bs4 import BeautifulSoup

# anlamsız kelimeler
STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn","also","would","could","may","might","shall","get","us","well",
    "one","two","three","new","use","used","using","work","worked","working",
    "experience","years","year","job","position","company","team","ability",
    "skills","skill","knowledge","strong","looking","responsible","provide",
    "required","requirements","preferred","including","including","must","need",
    "please","apply","equal","opportunity","employer","www","http","https",
    "com","org","net","edu","co","ltd","inc","llc",
}


def clean_html(text):
    # html etiketlerini temizler
    if not isinstance(text, str) or not text.strip():
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def clean_text(text):
    # küçük harfe çevir, noktalama işaretlerini sil
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_lemmatize(word):
    # kelimeleri köküne indirir
    suffixes = [("izations", "ize"), ("isation", "ize"), ("ization", "ize"),
                ("nesses", "ness"), ("ments", "ment"), ("ities", "ity"),
                ("ings", "ing"), ("tion", "tion"), ("sion", "sion"),
                ("ers", "er"), ("ies", "y"), ("ed", ""), ("es", ""),
                ("ing", ""), ("ly", ""), ("er", ""), ("s", "")]
    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[:-len(suffix)] + replacement
    return word


def remove_stopwords_and_lemmatize(text):
    tokens = text.split()
    tokens = [simple_lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def parse_list_field(value):
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            items = []
            for item in parsed:
                if isinstance(item, list):
                    items.extend([str(i) for i in item if i and str(i) != "None"])
                elif item and str(item) != "None":
                    items.append(str(item))
            return " ".join(items)
    except Exception:
        pass
    return value


def build_resume_text(row):
    parts = []
    objective = row.get("career_objective", "")
    if isinstance(objective, str) and objective.strip():
        parts.append(objective)
    for field in ["skills", "positions", "related_skils_in_job"]:
        val = parse_list_field(row.get(field, ""))
        if val:
            parts.append(val)
    responsibilities = row.get("responsibilities", "")
    if isinstance(responsibilities, str) and responsibilities.strip():
        parts.append(responsibilities)
    return " ".join(parts)


def build_job_text(row):
    parts = []
    position = row.get("position", "")
    if isinstance(position, str) and position.strip():
        parts.append(position)
    description = clean_html(row.get("description", ""))
    if description:
        parts.append(description)
    return " ".join(parts)


def preprocess_pipeline(text):
    text = clean_html(text)
    text = clean_text(text)
    text = remove_stopwords_and_lemmatize(text)
    return text
