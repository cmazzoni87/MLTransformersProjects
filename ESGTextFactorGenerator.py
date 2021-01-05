import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import itertools
import os
PATH_DOC = os.path.join(os.path.dirname(__file__), 'Documents')
PATH_OUT = os.path.join(os.path.dirname(__file__), 'Output')


def multi_run_wrapper(args):
    return max_sum_sim(*args)


def max_sum_sim(args):  # doc_embedding, word_embeddings, words, top_n, nr_candidates):
    # Calculate distances and extract keywords
    doc_embedding, word_embeddings, words, top_n, nr_candidates = args[0], args[1], args[2], args[3], args[4]
    distances = cosine_similarity(doc_embedding, word_embeddings)
    distances_candidates = cosine_similarity(word_embeddings,
                                            word_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim
    return [words_vals[idx] for idx in candidate]


def remove_non_alpha(srt):
    return re.sub(r'[^A-Za-z., ]', '', srt)


if __name__ == '__main__':
    generated_corpora_path = os.path.join(PATH_OUT, 'generated_corpora.csv')
    esg_taxonomy_factors = os.path.join(PATH_DOC, 'esg_taxonomy_factors.csv')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # add the EOS token as PAD token to avoid warnings
    model_gpt = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    key_words = {}
    data = pd.read_csv(esg_taxonomy_factors)[['FACTOR', 'Description']]
    data['Description'] = data['Description'].apply(remove_non_alpha)
    top_n = 40
    q = 0
    with open(generated_corpora_path, 'w', encoding='utf-8') as f1:
        f1.write('{0}|{1}\n'.format('Factor', 'Generated Text'))
        factors = list(set(data['FACTOR'].to_list()))
        countdown = len(factors)
        for factor in factors:
            print(factor)
            key_words[factor] = []
            descriptions = data[data['FACTOR'] == factor]['Description'].tolist()
            compiled_doc = '. '.join(descriptions)
            if len(compiled_doc.split(' ')) > 20:
                keyword = compiled_doc
                print(compiled_doc)
                # the more token the more the text will be equal to the sentences encoded
                input_ids = tokenizer.encode(keyword, return_tensors='tf', max_length=20)
                beam_norepeat_output = model_gpt.generate(input_ids,
                                                          # num_beams=100,
                                                          do_sample=True,
                                                          max_length=100,
                                                          min_length=70,
                                                          top_k=50,
                                                          top_p=0.95,
                                                          # temperature= 0.8,
                                                          no_repeat_ngram_size=2,
                                                          num_return_sequences=100,
                                                          # early_stopping=True
                                                          )

                for i, beam_output in enumerate(beam_norepeat_output):
                    generated_text = "{}".format(tokenizer.decode(beam_output, skip_special_tokens=True))
                    f1.write('{0}|{1}\n'.format(factor, remove_non_alpha(generated_text)))
                    print("Cleaned Output: {}".format(i))
                    print(generated_text)
                    print(100 * '=')
            q += 1
            print('{0} of {1} Done'.format(q, countdown))
    f1.close()
    clean_corpora = pd.read_csv(generated_corpora_path, delimiter='|')
    clean_corpora = clean_corpora.drop_duplicates()
    clean_corpora.to_csv(generated_corpora_path, sep='|', index=False)


