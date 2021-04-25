import requests
from os.path import dirname, abspath
from csv import reader

current_dir = dirname((abspath(__file__)))
data_path = current_dir+"/data/"


def read_qa_csv(csv_file):
    sequence_tokens_3 = {'question': [], 'answer': []}
    with open(csv_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                for i, block in enumerate(row):
                    if i == 0:
                        sequence_tokens_3['question'].append(block)
                    else:  # first question

                        sequence_tokens_3['answer'].append(block)
    return sequence_tokens_3


if __name__ == '__main__':
    url = "http://35.230.9.250:3389/inference"
    qa = read_qa_csv(data_path+"response_data.csv")
    for index, question in enumerate(qa["question"]):
        myobj = {'question': question}
        answer = requests.post(url, json=myobj)
        answer_dict = answer.json()
        print(answer_dict)
        print(qa["answer"][index])