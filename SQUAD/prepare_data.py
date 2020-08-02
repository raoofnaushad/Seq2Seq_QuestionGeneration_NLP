import os
import json
from tqdm import tqdm
import pandas as pd
import urllib.request
from collections import defaultdict

import config
from spacy.lang.en import English


tokenizer = English()
tokenizer.add_pipe(tokenizer.create_pipe("sentencizer"))

def clean_text(text):
    text = text.replace("]", " ] ")
    text = text.replace("[", " [ ")
    text = text.replace("\n", " ")
    text = text.replace("''", '" ').replace("``", '" ')
    return text

def sent_tokenize(text):
    return [[token.text for token in sentence if token.text] for sentence in tokenizer(text).sents]

def word_tokenize(text):
    tokens = [token.text for token in tokenizer(text) if token.text]
    tokens = [t for t in tokens if t.strip("\n").strip()]
    return tokens


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def prepare_csv():
    '''
        prepare a csv with four columns
        1) Context
        2) Sentence
        3) Question
        4) Answer
    '''
    for file_name in [config.SQUAD_TRAIN_FILE_NAME, config.SQUAD_DEV_FILE_NAME]:
        context_ll, sentence_ll, question_ll, answer_ll, sent_ans_ll = list(), list(), list(), list(),list()
        with open(os.path.join(config.SQUAD_OUTPUT_DIR, file_name), "r") as f:
            data = json.load(f)
        
        for article_id in (range(len(data["data"]))):
            list_paragraphs = data["data"][article_id]["paragraphs"]
            for paragraph in list_paragraphs:
                ## Each context paragraph
                context = paragraph['context']
                ## Cleaning text
                context = clean_text(context)
                ## Tokenizing words with spacy tokenizer
                context_tokens = word_tokenize(context)
                
                if (len(context_tokens) < config.min_len_context or len(context_tokens) > config.max_len_context):
                        continue
                ## Tokenizing sentence with spacy tokenizer
                context_sentences = sent_tokenize(context)
                ## span of tokens from the original text
                spans = convert_idx(context, context_tokens)
                num_tokens = 0
                first_token_sentence = []
                ## First token represent the first index token position of each sentence
                for sentence in context_sentences:
                    first_token_sentence.append(num_tokens)
                    num_tokens += len(sentence)  
                qas = paragraph['qas']
                for qa in qas:
                    question = qa['question']
                    question = clean_text(question)
                    question_tokens = word_tokenize(question)
                    if question_tokens[-1] != "?" or len(question_tokens) < config.min_len_question or len(question_tokens) > config.max_len_question:
                        continue
                    answer_ids = 1 if qa['answers'] else 0                    
                    
                    try:
                        answer = qa['answers'][0]['text']
                        answer = clean_text(answer)
                        answer_tokens = word_tokenize(answer)
                        answer_start = qa['answers'][0]['answer_start']
                        answer_stop = answer_start + len(answer)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_stop <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                
                        for idx, start in enumerate(first_token_sentence):
                            if answer_span[0] >= start:
                                sentence_tokens = context_sentences[idx]
                                answer_sentence_span = [span - start for span in answer_span]

                        context_ll.append(' '.join(tok for tok in context_tokens))
                        sentence_ll.append(' '.join(tok for tok in sentence_tokens))
                        question_ll.append(' '.join(tok for tok in question_tokens))
                        answer_ll.append(' '.join(tok for tok in answer_tokens))  
                        sent_ans = (' '.join(tok for tok in sentence_tokens)) + \
                           " answer " + (' '.join(tok for tok in answer_tokens))
                        sent_ans_ll.append(sent_ans)           
                    except:
                        pass
        dict = {'context':context_ll, 'sentence':sentence_ll, 'question':question_ll, \
            "answer":answer_ll, "sent_ans":sent_ans_ll}
        
        df = pd.DataFrame(dict)
        path_for_csv = os.path.join(config.SQUAD_OUTPUT_DIR, file_name.split('.')[0]+'.csv')
        df.to_csv(path_for_csv, index=False)
        print(df.head())
        print(df.shape)                


def download_squad(url, filename, out_dir):
    try:
        # path for local file.
        save_path = os.path.join(out_dir, filename)

        # check if the file already exists
        if not os.path.exists(save_path):
            # check if the output directory exists, otherwise create it.
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            print("Downloading", filename, "...")
            # download the dataset
            url = os.path.join(url, filename)
            file_path, _ = urllib.request.urlretrieve(url=url, filename=save_path)
        print("File downloaded successfully!")
    except:
        print("Some error occured!")
    
