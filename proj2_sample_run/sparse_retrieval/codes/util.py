import re
import pandas as pd


def read_title(query_path):
    with open(query_path, 'r', encoding='utf-8') as f:
        texts = f.read()
        qids = re.findall('<num> Number: (.*?)\n', texts)
        titles = re.findall('<title> (.*?)\n\n<desc>', texts)

    query = dict()
    for qid, title in zip(qids, titles):
        query[qid.strip()] = title

    return query


def read_topic(query_path):
    with open(query_path, 'r', encoding='utf-8') as f:
        texts = f.read()
        qids = re.findall('<num> Number: (.*?)\n', texts)
        descs = re.findall('<top>\n\n(.*?)\n</top>', texts, re.S)

    query = dict()
    for qid, desc in zip(qids, descs):
        query[qid.strip()] = desc

    return query


def read_desc(query_path):
    with open(query_path, 'r', encoding='utf-8') as f:
        texts = f.read()
        qids = re.findall('<num> Number: (.*?)\n', texts)
        descs = re.findall(r'<desc> Description:\s+(.*?)\n\n<narr>', texts, re.S)

    query = dict()
    for qid, desc in zip(qids, descs):
        query[qid.strip()] = desc

    return query


def read_title_desc(query_path):
    with open(query_path, 'r', encoding='utf-8') as f:
        texts = f.read()
        qids = re.findall('<num> Number: (.*?)\n', texts)
        titles = re.findall(r'<title> (.*?)\s+<desc>', texts, re.S)
        descs = re.findall(r'<desc> Description:\s+(.*?)\s+<narr>', texts, re.S)

    query = dict()
    for qid, title, desc in zip(qids, titles, descs):
        query[qid.strip()] = f"{title}\n{desc}"

    return query