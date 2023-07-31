from collections import Counter
import re

def extract_ngrams(x_raw, ngram_range=(1,3), token_pattern=r'\b[A-Za-z]{2,}\b', 
                   stop_words=[], vocab=set(), char_ngrams=False):
    ngrams = []
    tokens = [token.lower() for token in
             re.findall(token_pattern, x_raw) if
             token.lower() not in stop_words]
    
    if char_ngrams:
        charToken = []
        for i in range(len(tokens)):
            charToken.append([char for char in tokens[i]])

        tokens = [char for subCharTokens in
                       charToken for char in subCharTokens]

    
    for i in range(ngram_range[0], ngram_range[1]+1):
        if i == 1:
            ngrams += tokens
        else:
            ngrams += zip(*(tokens[j:] for j in range(i)))
        
    return [ngram for ngram in ngrams if ngram in vocab] if vocab else ngrams

def get_vocab(X_raw, ngram_range=(1,3), token_pattern=r'\b[A-Za-z]{2,}\b', 
              min_df=0, keep_topN=0, 
              stop_words=[],char_ngrams=False):
    
    df = Counter()
    ngram_counts = Counter()
    
    for t in X_raw:
        ngramExtracts = extract_ngrams(t, ngram_range, token_pattern,
                                      stop_words,
                                      char_ngrams)
        df.update(set(ngramExtracts))
        ngram_counts.update(ngram for ngram in ngramExtracts if df[ngram]>=min_df)
        
    vocab = {ngram for ngram, _ in ngram_counts.most_common(keep_topN)}
    
    return vocab, df, ngram_counts

def vectorise(X_ngram, vocab):
    X_vec = []
    for ngrams in X_ngram:
        c = Counter(ngrams)
        X_vec.append([c[v] for v in vocab])
    
    return X_vec
