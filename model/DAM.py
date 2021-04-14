import os
import time
import tensorflow as tf

# disable tensorflow debug information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os.path import dirname, abspath
current_dir = dirname((abspath(__file__)))


from .utils import net, predict, preprocessor,generate_data

#
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

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

def load_model():
    # define model class
    model = net.Net(conf)
    graph = model.build_graph()
    print('build graph sucess during load model')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    sess = tf.compat.v1.Session(graph=graph)
    with graph.as_default():
        with sess.as_default():
            # _model.init.run();
            # _model.saver = tf.train.import_meta_graph("init_meta")
            print(f'tf.compat.v1.get_default_session():{tf.compat.v1.get_default_session()}')
            model.saver = tf.compat.v1.train.import_meta_graph(conf["init_meta"])
            print(f'model after load graph:{model.saver}')
            model.saver.restore(sess, conf["init_model"])
            print(f'model after restore:{model.saver}')
            print("sucess init %s" % conf["init_model"])
    print(f'sess out of with:{sess}')
    print(tf.compat.v1.get_default_session())
    return model,graph,sess


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


def prepare_q_a_data(question_number,cls_indexs, question_text, answers_text,word_dict,key_words_list):
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

    return all_data,text_data_classified
    # return indexs,all_data

def find_question_answer(question, question_text,answers_text):
    for index,value in enumerate(question_text):
        if question in value:
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


def build_qa_with_bilistm(question, answers):
    q_a_set = []
    for answer in answers:
        q = question
    #    q.append('\t')
        q_a = q + '\t'+ answer
        positive_flag = '1' + '\t'
        q_a = positive_flag + q_a
        print(q_a)
      #  q_a.insert(0, positive_flag)
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


def model_interface(input,graph,model,sess):
    print("enter the interface method")
    split_input = input.split("___SEP___")
    if len(split_input)==1:
        SINGLEMODEL = True
    else:
        SINGLEMODEL = False
    print(split_input)
    print(SINGLEMODEL)
    return dam_output(split_input,SINGLEMODEL,graph,model,sess)


# Customize your model logic here. Feel free to change the function name.
# Customize your model logic here. Feel free to change the function name.
def dam_output(split_input,SINGLEMODEL,graph,model,sess):
    # # define model class
    # model = net.Net(conf)

    # if no bilstm, should work out with the proposed answer by itself; otherwise, get the proposed answers from bilstm
    if SINGLEMODEL:
        input = split_input
        key_words_list = ["input classification", "output", "context"]
        cls_indexs, question_text, answers_text,word_dict = prepare_data(data_path)
        for number, question in enumerate(question_text):
            if number == 0:
                print(question[0])
                print(type(question[0]))
            if input in question:
                print(f'find the question equal to input:{input}')
                print(f' the found number is: {number}')
                print(f'the question is: {question}')
                break
        print(f'the input question type is {type(input)}')
        print(f'the input question is: {input}')
        print(f"find the quesiton number is: {number}")
        print(f'the question is: {question_text[number]}')
        question_number = [number]
        all_data,text_data_classified = prepare_q_a_data(question_number,cls_indexs, question_text, answers_text,word_dict,key_words_list)
        indexs, answers = predict.test_with_model(conf,  model,graph,sess, text_data_classified)
        print(indexs)
        output = pop_answers(indexs,question_text,question_number,all_data)
    else:
        question = split_input[0]
        answers = split_input[1].split("__EOS__")
        cls_indexs, question_text, answers_text, word_dict = prepare_data(data_path)
        print(f'question is:{question}')
        print("the candidate answers are:\n")
        print(answers)
        # questions = question
        q_a_set = build_qa_with_bilistm(question, answers)
        text_data_classified = preprocessor.get_sequence_tokens_with_turn(q_a_set, word_dict)
        indexs, answers = predict.test_with_model(conf,  model, graph,sess,text_data_classified)
        print(indexs)
        print(f'answer is: {this_answer}')
        output = answers
    return output


if __name__ == '__main__':
    test_cls_indexs, test_question_text, test_answers_text, word_dict = prepare_data(data_path)
    question = "add nde___SEP___What do you mean by property?What do you mean by property?__EOS__The order of nodes at each level will be automatically sorted based on name (A-Z) after each action. The order doesn't affect test cases generation.__EOS__No. This feature will need to be developed.__EOS__There is no limit to the number of nodes you can create at each level. You can add nodes as needed__EOS__Do you want to zoom in or zoom out the tree?__EOS__Sure. You can choose to upload files or create online.__EOS__There is no limit. You can add child nodes as needed. \n__EOS__No.Also User should not use same login with currently with multiple browser.__EOS__No. Currently we do not have this facility.__EOS__There is no such limitationn."
    print(question)
    model,graph,sess=load_model()
    model_interface(question,model,graph,sess)
