# import cPickle as pickle
import pickle
import numpy as np
import os
import math

def unison_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.array(data['y'])
    c = np.array(data['c'])
    r = np.array(data['r'])

    assert len(y) == len(c) == len(r)
    p = np.random.permutation(len(y))
    shuffle_data = {'y': y[p], 'c': c[p], 'r': r[p]}
    return shuffle_data

def split_c(c, split_id):
    '''c is a list, example context
       split_id is a integer, conf[_EOS_]
       return nested list
    '''
    turns = [[]]
    for _id in c:
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns

def normalize_length(_list, length, cut_type='tail'):
    '''_list is a list or nested list, example turns/r/single turn c
       cut_type is head or tail, if _list len > length is used
       return a list len=length and min(read_length, length)
    '''
    real_length = len(_list)
    if real_length == 0:
        return [0]*length, 0

    if real_length <= length:
        if not isinstance(_list[0], list):
            _list.extend([0]*(length - real_length))
        else:
            _list.extend([[]]*(length - real_length))
        return _list, real_length

    if cut_type == 'head':
        return _list[:length], length
    if cut_type == 'tail':
        return _list[-length:], length

def produce_one_sample(data, index, split_id, max_turn_num, max_turn_len, turn_cut_type='tail', term_cut_type='tail'):
    '''max_turn_num=10
       max_turn_len=50
       return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len
    '''
    c = data['c'][index]
    r = data['r'][index][:]
    y = data['y'][index]

    turns = split_c(c, split_id)
    #normalize turns_c length, nor_turns length is max_turn_num
    nor_turns, turn_len = normalize_length(turns, max_turn_num, turn_cut_type)

    nor_turns_nor_c = []
    term_len = []
    #nor_turn_nor_c length is max_turn_num, element is a list length is max_turn_len
    for c in nor_turns:
        #nor_c length is max_turn_len
        nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type)
        nor_turns_nor_c.append(nor_c)
        term_len.append(nor_c_len)

    nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)

    return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len

def build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns = []
    _tt_turns_len = []
    _every_turn_len = []

    _response = []
    _response_len = []

    _label = []
    ## after this step, all the turns in one c are normalized to maximal length; all the c after normalization are append together.
    for i in range(conf['batch_size']):
        index = batch_index * conf['batch_size'] + i
        y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len = produce_one_sample(data, index, conf['_EOS_'], conf['max_turn_num'],
                conf['max_turn_len'], turn_cut_type, term_cut_type)

        _label.append(y)
        _turns.append(nor_turns_nor_c)
        _response.append(nor_r)
        _every_turn_len.append(term_len)
        _tt_turns_len.append(turn_len)
        _response_len.append(r_len)
    ## each batch data consists of batch size lists. Each element in this list is normalized list with normalized elements
    return _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label

def build_one_batch_dict(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label = build_one_batch(data, batch_index, conf, turn_cut_type, term_cut_type)
    ans = {'turns': _turns,
            'tt_turns_len': _tt_turns_len,
            'every_turn_len': _every_turn_len,
            'response': _response,
            'response_len': _response_len,
            'label': _label}
    return ans
    

def build_batches(data, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns_batches = []
    _tt_turns_len_batches = []
    _every_turn_len_batches = []

    _response_batches = []
    _response_len_batches = []

    _label_batches = []

    print(len(data['y']))
    print(conf['batch_size'])

    batch_len = int(len(data['y'])/conf['batch_size'])
    for batch_index in range(batch_len):
        _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label = build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail')

        _turns_batches.append(_turns)
        _tt_turns_len_batches.append(_tt_turns_len)
        _every_turn_len_batches.append(_every_turn_len)

        _response_batches.append(_response)
        _response_len_batches.append(_response_len)

        _label_batches.append(_label)

    ans = { 
        "turns": _turns_batches, "tt_turns_len": _tt_turns_len_batches, "every_turn_len":_every_turn_len_batches,
        "response": _response_batches, "response_len": _response_len_batches, "label": _label_batches
    }   

    return ans 

if __name__ == '__main__':
    data_path = "/Users/ally/Documents/12020Fall/data298/github/chatbot/data/"

    conf = { 
        "batch_size": 64,
        "max_turn_num": 10, 
        "max_turn_len": 50, 
        "_EOS_": 28270,
        "train_path": data_path + "train.pickle",
        "valid_path": data_path + "valid.pickle",
        "test_path": data_path + "test.pickle",
    }
    train = pickle.load(open(conf["train_path"],'rb'))
    val = pickle.load(open(conf["valid_path"],'rb'))
    test = pickle.load(open(conf["test_path"],'rb'))

#    train, val, test = pickle.load(open(data_file, 'rb'))
    print('load data success')

    # data batch includes below information:
    # turns: the c parts, list of batch, each batch consists batch size list, in each list is the normalized turns
    # tt_turns_len: corresponding to turns, consists the turn number  information of tokenized c
    # every_turn leng: every turn length information
    # response: normalized repsonse in batches. list of list. the first layer is each batch
    # repsonse_len: each response tokenized length
    # label: each y corresponding to c and r

    train_batches = build_batches(train, conf)
    val_batches = build_batches(val, conf)
    test_batches = build_batches(test, conf)
    print('build batches success')
    
    pickle.dump([train_batches, val_batches, test_batches], open('../../data/batches_small.pkl', 'wb'))
    print('dump success')


        


    








    
    


