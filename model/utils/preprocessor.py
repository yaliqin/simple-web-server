import os
import glob
import json
import csv
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence


def raw_data_parser(data_tsv):
    # input: raw data
    # output: corpus list[str]
    corpus = []
    with open(data_tsv, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
        record = ''
        for i, row in enumerate(reader):
            if row[1].upper() == "U" or row[1].upper() == "R":
                record += ('\t' + row[2])
            elif row[1].lower() == "flag":
                corpus.append(row[2] + record)
                record = ''
            else:
                continue
    return corpus

def _raw_data_parser(data_tsv):
    corpus = []
    with open(data_tsv, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', quotechar='|')
        record = ''
        for i, row in enumerate(reader):
            if i == 0:
                continue
            record += (row[4] + '\t')
            sentences = row[2].split('__EOS__')
            context = "\t".join(sentences)
            record += (context + '\t' + row[3])
            corpus.append(record)
            record = ''
    return corpus

def dump_corpus_to_txt(corpus, data_tsv):
    # input: corpus list[str]
    # output: tab delimited txt file

    filename = os.path.splitext(data_tsv)[0] + '.txt'
    with open(filename, "w") as f:
        for sentence in corpus:
            print(sentence, file=f)

def _merge_multiple_txt(file_path):
    filenames = []
    for filename in glob.glob(os.path.join(file_path, '*.txt')):
        filenames.append(filename)
    with open('all_data.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def read_txt_file(data_txt):
    # input: tab delimited txt file
    # output: corpus list[str]
    corpus = []
    with open(data_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            corpus.append(line)
    return corpus

def get_texts(corpus):
    texts = []
    for line in corpus:
        for i, block in enumerate(line.split('\t')):
            if i == 0:
                continue
            texts.append(block)
    return texts

def generate_word_dict(texts):
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(texts)
    return tokenizer.word_index

def dump_word_dict_to_json(word_dict):
    with open('word_dict.json', 'w') as jsonFile:
        json.dump(word_dict, jsonFile)

def _get_tokens(texts):
    tokens = []
    for line in texts:
        tokens.append(text_to_word_sequence(line))
    return tokens

def _word_to_sequence(tokens, word_dict):
    for line in tokens:
        for i, word in enumerate(line):
            line[i] = word_dict[word]
    return tokens

def get_sequence_tokens(corpus, word_dict):
    sequence_tokens = {'y':[], 'c':[], 'r':[]}
    for line in corpus:
        blocks = line.split('\t')
        context = []
        for i, block in enumerate(blocks):
            if i == 0:
                sequence_tokens['y'].append(int(block))
            else:
                tokens = text_to_word_sequence(block)
                for j, word in enumerate(tokens):
                    tokens[j] = word_dict[word]
                if i == len(blocks) - 1:
                    sequence_tokens['r'].append(tokens)
                else:
                    context.extend(tokens)
        sequence_tokens['c'].append(context)
    return sequence_tokens


def get_sequence_tokens_with_turn(corpus, word_dict):
# this function generate dataset as 'c','r','y', the multiple turns are split with 28270: __EOS__ to align with reader.py
    sequence_tokens = {'y':[], 'c':[], 'r':[]}
    for line in corpus:
        blocks = line.split('\t')
        context = []
        for i, block in enumerate(blocks):
            if i == 0:
                sequence_tokens['y'].append(int(block))
            elif i == len(blocks) - 1:
                context.pop(-1)
                sequence_tokens['c'].append(context)
                tokens = text_to_word_sequence(block)
                for j, word in enumerate(tokens):
                    tokens[j] = word_dict[word]
                sequence_tokens['r'].append(tokens)
            else:
                tokens = text_to_word_sequence(block)
                for j, word in enumerate(tokens):
                    tokens[j] = word_dict[word]
                context.extend(tokens)
                context.append(28270)

    return sequence_tokens

def generate_train_valid_test_data(sequence_tokens):
    # input: sequence_tokens dictionary
    # output: tuple of sequence_tokens dictionary, namely training, validation and test set
    pass

def dump_data_to_pkl(data, filename):
    # input: tuple of ready data
    # output: pkl format ready data
    filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    corpus = read_txt_file("../data/original_data2.txt")
    texts = get_texts(corpus)
    word_count = 0
    dialog_count = 0
    for line in texts:
        dialog_count +=1
        for text in line:
            word_count +=1
    print(word_count)
    print(dialog_count)
    # word_dict = generate_word_dict(texts)
    # dump_word_dict_to_json(word_dict)
    # sequence_tokens = get_sequence_tokens(corpus, word_dict)
    # dump_data_to_pkl(sequence_tokens, 'all_data')
    # with open('original_data.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
    # print(b)


    # for filename in glob.glob(os.path.join(file_path, '*.txt')):
    #     corpus = raw_data_parser(filename)
    #     dump_corpus_to_txt(corpus, filename)