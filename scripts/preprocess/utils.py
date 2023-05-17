from typing import List, Tuple
from numpy import ndarray
import re


#--- Compile regex ----------------------------------------

PATTERNS = [
    r"\[.+\]" # match anything within brackets
]
REGEXES = []
for p in PATTERNS:
    REGEXES.append(re.compile(p))

#--- Util functions ---------------------------------------

def preprocess_context(context: str, direction: str) -> str:
    l = 1 if direction == "left" else 0

    for r in REGEXES:
        context = r.sub("", context)
    if "." in context:
        context = context.split(".")[l]
    if "\n" in context:
        context = context.split("\n")[l]

    if direction=="left": context = context.lstrip()
    else: context = context.rstrip()

    return context


def get_mention_context(
        raw_text: str, span: List[int], spans,# indicator: ndarray, FIXME
        context_len: int = 10
    ) -> Tuple:
    '''Formats a mention with its surrounding context for training
    ---
    raw_text : raw source txt for mention
    span : [start, end] character span in raw source txt
    indicator : an integer array that indicates whether a character belongs
        to a named entity (e.g. CELL, BIOLOGICAL_PROCESS)
    context_len : number of contextual words to wrap around mention
    ---
    returns list of ["left_context", "mention", "right_context"]
    '''
    # left
    left_words = raw_text[:span[0]].split(" ")
    if len(left_words) == 0:
        left_context = ""
    elif len(left_words) <= context_len:
        left_context = " ".join(left_words)
        left_context = preprocess_context(left_context, "left")
    else:
        left_context = " ".join(left_words[-context_len:])
        left_context = preprocess_context(left_context, "left")
        if "." in left_context:
            left_context = left_context.split(".")[1]

    # mention
    mention = raw_text[span[0]:span[1]]

    # right
    right_words = raw_text[span[1]:].split(" ")
    if len(right_words) == 0:
        right_context = ""
    elif len(right_words) <= context_len:
        right_context = " ".join(right_words)
        right_context = preprocess_context(right_context, "right")
        if "." in right_context:
            right_context = right_context.split(".")[0]
    else:
        right_context = " ".join(right_words[:context_len])
        right_context = preprocess_context(right_context, "right")
        if "." in right_context:
            right_context = right_context.split(".")[0]

    # get spans within the local of the mention
    m_spans = []
    for s in spans:
        if (s[1] <= span[1] + context_len) & (s[0] >= span[0] - context_len):
            m_spans.append(s)

    return (left_context, mention, right_context), m_spans
