import sys
import os

# disable tensorflow debug information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os.path import dirname, abspath
current_dir = dirname((abspath(__file__)))

from utils import net, predict, preprocessor,generate_data


sys.path.append("../../")




home_folder = "/home/ally/github/chatbot/"
result_path = home_folder+"models/DAM/results/"
data_path = current_dir+"/data/"
model_folder = current_dir+"/saved_model/"

# data_path =  home_folder+"data/"
conf = {
    "data_path": data_path+"classified_split.pickle",
    "train_path":data_path+"classified_train.pickle",
    "valid_path":data_path+"classified_valid.pickle",
    "test_path":data_path+"classified_test.pickle",
    "word_emb_init":None,

    "save_path": result_path,
    #"init_model":"/home/ally/DAM/output/ubuntu/DAM/DAM.ckpt",
    #"init_meta":"/home/ally/DAM/output/ubuntu/DAM/DAM.ckpt.meta",
    "init_model":model_folder+"model.ckpt.21.0", # for local machine test
    "init_meta":model_folder+"model.ckpt.21.0.meta",

    "rand_seed": None,

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,

    "stack_num": 5,
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "vocab_size": 9449,
    "emb_size": 200,
    "batch_size": 10, #200 for test

    "max_turn_num": 9,
    "max_turn_len": 50,

    "max_to_keep": 1,
    "num_scan_data": 2,
    "_EOS_": 28270, #1 for douban data
    "final_n_class": 1,
}


def prepare_data(data_path):
    data_file = data_path+"all_classified_data.txt"
    corpus = preprocessor.read_txt_file(data_file)
    texts = preprocessor.get_texts(corpus)
    word_dict = preprocessor.generate_word_dict(texts)
    answers_text= []
    question_text = []
    positive_corpus =[]
    for item in corpus:
        blocks = item.split('\t')
        if(blocks[0]=='1'):
            answers_text.append(([blocks[-1]]))
            question_text.append((blocks[1:-1]))
            positive_corpus.append(item)
    cls_indexs, question_text, answers_text = generate_data.get_subset_answers(data_path)
    # print(cls_indexs)
    return cls_indexs, question_text, answers_text,word_dict


def prepare_q_a_data(question_number,cls_indexs, question_text, answers_text,word_dict,key_words_list,model):
    all_data = []
    for index in question_number:
        #    print(f'the {index} question is:{question_text[index]}')
        question = question_text[index]
        positive_answer, negative_answers, negative_answers_index = \
            generate_data.generate_candidate_answers(question,key_words_list,cls_indexs, question_text, answers_text)
        negative_answers_index.insert(0, index)
        all_data.append(positive_answer[0])
        #    print(positive_answer[0])
        for item in negative_answers:
            all_data.append(item)
        # print(all_data)
    text_data_classified = preprocessor.get_sequence_tokens_with_turn(all_data, word_dict)
    indexs, answers = predict.test(conf, model, text_data_classified)
    print(indexs)
    return indexs,all_data

def find_question_answer(question, question_text,answers_text):
    for index,value in enumerate(question_text):
        if question == value:
            select_answer = answers_text[index]
            return index, select_answer


def build_bilstm_qa(questions,question_text,answers_text):
    q_a_set =[]
    for q in questions:
        index, a = find_question_answer(q, question_text,answers_text)
        q.append('\t')
        q_a = q + a
        positive_flag = '1' + '\t'
        q_a.insert(0, positive_flag)
        q_a_set.append(q_a)
    return q_a_set




def pop_answers(indexs,question_text,question_number,all_data):
    ind = 0
    answers =[]
    for index, number in enumerate(question_number):
        print(f'question number is: {number}')
        print(f'question is: {question_text[number]}')
        print(f'answer index is {indexs[index]} in the classification list')
        idx_in_all = ind * 10 + indexs[index]
        print(idx_in_all)
        answer_data = all_data[idx_in_all]
        this_answer = answer_data.split('\t')[-1]
        print(f'anwer is: {this_answer}')
        ind += 1
        answers.append(this_answer)
    return answers
    #


def model_interface(input):
    SINGLEMODEL = 1
    return dam_output(input,SINGLEMODEL)


# Customize your model logic here. Feel free to change the function name.
# Customize your model logic here. Feel free to change the function name.
def dam_output(input,SINGLEMODEL):
    # define model class
    model = net.Net(conf)

    # if no bilstm, should work out with the proposed answer by itself; otherwise, get the proposed answers from bilstm
    if SINGLEMODEL == 1:
        key_words_list = ["input classification", "output", "context"]
        cls_indexs, question_text, answers_text,word_dict = prepare_data(data_path)
        for number, question in enumerate(question_text):
            if question == input:
                break
        question_number = [number]
        indexs,all_data = prepare_q_a_data(question_number,cls_indexs, question_text, answers_text,word_dict,key_words_list,model)
        output = pop_answers(indexs,question_text,question_number,all_data)
    else:
        cls_indexs, question_text, answers_text, word_dict = prepare_data(data_path)
        print(f'question is:{input}')
        questions = input
        q_a_set = build_bilstm_qa(questions, question_text, answers_text)
        text_data_classified = preprocessor.get_sequence_tokens_with_turn(q_a_set, word_dict)
        indexs, answers = predict.test(conf, model, text_data_classified)
        answer_data = q_a_set[indexs]
        this_answer = answer_data.split('\t')[-1]
        print(f'answer is: {this_answer}')
    return output


if __name__ == '__main__':
    test_cls_indexs, test_question_text, test_answers_text, word_dict = prepare_data(data_path)
    question = test_question_text[22]
    model_interface(question)
