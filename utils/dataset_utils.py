import numpy as np
from rich.progress import Progress
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn,TimeElapsedColumn

def get_abstracts_dict(path):
    abstracts = dict()
    with open(path, 'r', encoding="latin-1") as f:
        for line in f:
            node, full_abstract = line.split('|--|')
            abstract_list = full_abstract.split('.')
            full_abstract = '[SEP]'.join(abstract_list)

            abstracts[int(node)] = full_abstract
    return abstracts

def get_authors_dict(path):
    authors = dict()
    total_authors = []
    with open(path, 'r', encoding="latin-1") as f:
        for line in f:
            node, full_authors = line.split('|--|')
            authors_list = full_authors.replace('\n','').split(',')
            authors[int(node)] = authors_list
            total_authors.extend(authors_list)
    unique_authors = np.unique(np.array(total_authors))
    return authors, unique_authors

def get_progress_bar():
    return Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("[bold blue]{task.fields[info]}", justify="right"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        "\n"
    )