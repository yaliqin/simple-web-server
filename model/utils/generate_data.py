from . import preprocessor
import pickle
from random import sample


def get_ones(lista):
  ones=0
  l = len(lista)
  for item in lista:
    ones+=item
  print(ones/l)

def data_split(data_file, train_ratio, val_ratio):
  corpus = preprocessor.read_txt_file(data_file)
  l_total = len(corpus)
  l_train = int(train_ratio * l_total)
  l_val = int(val_ratio * l_total)
  train_data = corpus[:l_train]
  val_data = corpus[l_train:l_train + l_val]
  test_data = corpus[l_train + l_val:]
  return(corpus, train_data,val_data,test_data)

def get_subset(positive_corpus,positive_answers, keywords):
  answer_subset = []
  index_subset = []
  for index, item in enumerate(positive_corpus):
    if keywords in item:
      answer_subset.append(positive_answers[index])
      index_subset.append(index)
  return answer_subset,index_subset


def write_file(file_path, var):
  with open(file_path, 'w') as f:
    for item in var:
      line = item[0]
      f.write(line)
    f.close()

def read_file(data_file):
  corpus = []
  with open(data_file, "r") as f:
    lines = f.readlines()
    for line in lines:
      corpus.append(line)
  return corpus

def get_subset_answers(data_path):
  data_file = data_path + "original_data2.txt"
  corpus = preprocessor.read_txt_file(data_file)
  # texts = preprocessor.get_texts(corpus)
  # word_dict = preprocessor.generate_word_dict(texts)
  answers_text = []
  question_text = []
  positive_corpus = []
  for item in corpus:
    blocks = item.split('\t')
    if (blocks[0] == '1'):
      answers_text.append(([blocks[-1]]))
      question_text.append((blocks[1:-1]))
      positive_corpus.append(item)
  input_classification_answers,input_cls_index = get_subset(positive_corpus,answers_text,'input classification')
  output_classification_answers,output_cls_index = get_subset(positive_corpus,answers_text,'output')
  context_classification_answers,contex_cls_index = get_subset(positive_corpus,answers_text,'contex')
  orset_index = input_cls_index + output_cls_index + contex_cls_index
  corpus_index = list(range(len(positive_corpus)))
  process_cls_index = [n for n in corpus_index if n not in orset_index]
  process_answers = []

  for index in process_cls_index:
    process_answers.append(answers_text[index])


  input_classification_file = data_path + "input_classification_answers.txt"
  output_classification_file = data_path + "output_classification_answers.txt"
  context_classification_file = data_path + "context_classification_answers.txt"
  process_file = data_path + "process_answers.txt"
  write_file(input_classification_file, input_classification_answers)
  write_file(output_classification_file, output_classification_answers)
  write_file(context_classification_file,context_classification_answers)
  write_file(process_file, process_answers)
  cls_indexs = [input_cls_index, output_cls_index, contex_cls_index,process_cls_index]
  return cls_indexs, question_text, answers_text

  # question_number = [1, 10]
  # all_positive_answers = predict.build_candidate_answers(positive_corpus, word_dict)

def generate_candidate_answers(question, key_words_list, cls_indexs,question_text, answers_text):
  # question: the current question
  # key_words_list: input_classification, output_classification, context, process
  # cls_indexs: the index of each classification problems, 4 elements list, each element is a list containing the index
  # question_text: all questions
  # answer_text: all questions
  question_combine = ['\t'.join(question)]

  for index, key_word in enumerate(key_words_list):
    if key_word in question_combine:
      answers_index = cls_indexs[index]
      break
    if index == 2: #belong to process questions
      answers_index = cls_indexs[3]
  question_index = question_text.index(question)
  print(question_index)
  negative_data_length = 9
  negative_index_list = [n for n in answers_index if n != question_index]
  negative_indexs = sample(negative_index_list, negative_data_length)

  # question = ['\t'.join(question)]
  question_combine.append('\t')
  flag = '0' + '\t'

  negative_answer = []
  positive_answer = question_combine + answers_text[question_index]
  positive_flag = '1' +'\t'
  positive_answer.insert(0,positive_flag)
  positive_answer=[''.join(positive_answer)]
  negative_answers = []
  for num in negative_indexs:
    negative_answer = answers_text[num]
    s = question_combine + negative_answer
    s.insert(0, flag)
    s1 = [''.join(s)]

    negative_answers.append(s1[0])
  return positive_answer, negative_answers,negative_indexs

def generate_all_candidate_answers(question, key_words_list, cls_indexs,question_text, answers_text):
  # question: the current question
  # key_words_list: input_classification, output_classification, context, process
  # cls_indexs: the index of each classification problems, 4 elements list, each element is a list containing the index
  # question_text: all questions
  # answer_text: all questions
  question_combine = ['\t'.join(question)]

  for index, key_word in enumerate(key_words_list):
    if key_word in question_combine:
      answers_index = cls_indexs[index]
      break
    if index == 2: #belong to process questions
      answers_index = cls_indexs[3]
  question_index = question_text.index(question)
  print(question_index)
  # negative_data_length = 9
  # negative_index_list = [n for n in answers_index if n != question_index]
  # negative_indexs = sample(negative_index_list, negative_data_length)

  negative_indexs = negative_index_list


  # question = ['\t'.join(question)]
  question_combine.append('\t')
  flag = '0' + '\t'

  negative_answer = []
  # positive_answer = question_combine + answers_text[question_index]
  # positive_flag = '1' +'\t'
  # positive_answer.insert(0,positive_flag)
  # positive_answer=[''.join(positive_answer)]
  negative_answers = []
  for num in negative_indexs:
    negative_answer = answers_text[num]
    s = question_combine + negative_answer
    s.insert(0, flag)
    s1 = [''.join(s)]

    negative_answers.append(s1[0])
  return negative_answers,negative_indexs



  # for index, question in corpus:
  #   negative_data_length = 9
  #   negative_index_list = [n for n in index_list if n != index]
  #   negative_indexs = sample(negative_index_list, negative_data_length)
  #   for num in negative_indexs:
  #     question = question_text[index]
  #     question = ['\t'.join(question)]
  #     question.append('\t')
  #     flag = '0' + '\t'
  #     negative_answer = answers_text[num]
  #     s = question + negative_answer
  #     s.insert(0, flag)
  #     s1 = [''.join(s)]
  #     new_data.append(s1[0])
  #
  # new_data_path = data_path + "new_data_10answers.txt"
  # with open(new_data_path, 'w') as f:
  #   for item in new_data:
  #     line = str(item)
  #     f.write(line)
  #   f.close()




def generate_train_data(key_words_list,cls_indexs, question_text, answers_text):
  # this function is generate the train data for 1:9 positive and negative answers
  all_data = []
  for index, question in enumerate(question_text):
    positive_answer, negative_answers = generate_candidate_answers(question, key_words_list, cls_indexs, question_text, answers_text)
    all_data.append(positive_answer[0])
    for item in negative_answers:
      all_data.append(item)

  return all_data




if __name__ == '__main__':
  data_path = "../data/"
  cls_indexs, question_text, answers_text =  get_subset_answers(data_path)
  key_word_lists = ["input classification", "output", "context"]

  all_data = generate_train_data(key_word_lists,cls_indexs, question_text, answers_text)
  all_data_file = data_path + "all_classified_data.txt"
  with open(all_data_file, 'w') as f:
    for item in all_data:
      line = item
      f.write("%s" % item)
    f.close()

  data_file = "../data/all_classified_data.txt"

  corpus, train, val, test = data_split(data_file, 0.7,0.2)
  texts = preprocessor.get_texts(corpus)


  word_dict = preprocessor.generate_word_dict(texts)

  train_sequence_tokens = preprocessor.get_sequence_tokens_with_turn(train, word_dict)
  val_sequence_tokens = preprocessor.get_sequence_tokens_with_turn(val, word_dict)
  test_sequence_tokens = preprocessor.get_sequence_tokens_with_turn(test, word_dict)
  data_tokens = {'train':[],'val':[],'test':[]}
  data_tokens['train']=train_sequence_tokens
  data_tokens['val']=val_sequence_tokens
  data_tokens['test']=test_sequence_tokens


  preprocessor.dump_data_to_pkl(data_tokens,data_path+'classified_data_split')
  preprocessor.dump_data_to_pkl(test_sequence_tokens,data_path+'classified_test')
  preprocessor.dump_data_to_pkl(train_sequence_tokens,data_path+'nclassified_train')
  preprocessor.dump_data_to_pkl(val_sequence_tokens,data_path+'classified_valid')
