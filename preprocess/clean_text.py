import pandas as pd
import re
from tqdm import tqdm

def splitSentence(content):
    """
    Given block of text, split into sentence
    Output: list of sentences
    """
    # Multiple space to single space, remove separators like - and _
    if pd.notnull(content):
        content = re.sub('\s*\t\t\t', ' ', content)
        content = re.sub('\n+', ' ', content)
        content = re.sub('--+|==+|__+|\*\*\*+', ' ', content)
        content = re.sub('\.\s+', '. ',content)
        content = re.sub(':\s+', ': ',content)
        content = re.sub('\s+\[\*', ' [*', content)
        content = re.sub(' \s+', ' ',content)
        lsS = content.split('. ')
    else:
        lsS = []
    return lsS


def update(s):
    """
    #- replace number to <num> (keep number right after text, as typically are certain clinical names)
    #- replace time to <time>
    - replace digits to <N> token
    - add space before/after non-character
    """
    s = re.sub('\d', 'N', s)
    #s = re.sub('\d+:\d+(:\d+)?\s*(((a|A)|(p|P))(m|M))?(\s*est|EST)?', ' <time> ', s)
    #s = re.sub('( |^|\(|:|\+|-|\?|\.|/)\d+((,\d+)*|(\.\d+)?|(/\d+)?)', ' <num> ', s) # cases like: 12,23,345; 12.12; .23, 12/12;
    s = re.sub(r'([a-zA-Z->])([<\),!:;\+\?\"])', r'\1 \2 ', s)
    s = re.sub(r'([\(,!:;\+>\?\"])([a-zA-Z<-])', r' \1 \2', s)
    s = re.sub('\s+', ' ', s)
    return s


def replcDeid(s):
    """
    replace de-identified elements in the sentence (date, name, address, hospital, phone)
    """
    s = re.sub('\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]', ' <date> ', s)
    s = re.sub('\[\*\*[^\]]*?Date[^\]]*?\*\*\]', ' <date> ', s)
    s = re.sub('\[\*\*[^\]]*?Name[^\]]*?\*\*\]', ' <name> ', s)
    s = re.sub('\[\*\*[^\]]*?(phone)[^\]]*?\*\*\]', ' <phone> ', s)
    s = re.sub('\[\*\*[^\]]*?(Hospital|Location|State|Address|Country|Wardname|PO|Company)[^\]]*?\*\*\]', ' <loc> ', s)
    s = re.sub('\[\*\*[^\]]*?\*\*\]', ' <deidother> ', s)
    return s


def cleanString(s, lower = True):
    s = replcDeid(s)
    s = update(s)
    if lower:
        s = s.lower()
    return s


def replaceContractions(s):
    contractions = ["don't","wouldn't","couldn't","shouldn't", "weren't", "hadn't" , "wasn't", "didn't" , "doesn't","haven't" , "isn't","hasn't"]
    for c in contractions:
        s = s.replace( c, c[:-3] +' not')
    return s

def preprocess_string(s):
    s = cleanString(s, True)
    s = replaceContractions(s)
    return s


def cleanNotes(content):
    """
    Process a chunk of text
    """
    lsOut = []
    content = str(content)
    if len(content) > 0:
        lsS = splitSentence(content)
        for s in lsS:
            if len(s) > 0:
                s = cleanString(s, lower = True)
                s = replaceContractions(s)
                lsOut.append(s)
        out = ' '.join(lsOut)
    else:
        out = ''
    return out

def preprocess(df_icu_notes):
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_info_columns', -1)
    old_text = []
    for i in range(1):
        old_text.append(df_icu_notes.iloc[i]['TEXT'])
    total = len(df_icu_notes.index)
    with tqdm(total=total) as pbar:
        for index, row in tqdm(df_icu_notes.iterrows()):
            pbar.update(1)
            text = row['TEXT']
            new_text = cleanNotes(text)
            df_icu_notes.loc[index, 'TEXT'] = new_text
    print('\nSample of note text after preprocessing')
    for i in range(1):
        print('\n\n\n\n\nOld Text: ')
        print(old_text[i])
        print('\n\nNew Text: ')
        print(df_icu_notes.iloc[i]['TEXT'])
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.max_info_columns', 100)
    return df_icu_notes
