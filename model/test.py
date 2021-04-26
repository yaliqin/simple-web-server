import requests
from os.path import dirname, abspath
from csv import reader,writer

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

def write_qa_csv(data,result_csv):
    fields = ['question', 'qa_anw', 'model_anw']
    with open(result_csv, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(data)


if __name__ == '__main__':
    url = "http://35.230.9.250:3389/inference"
    input_data = data_path+"response_data.csv"
    output_data =data_path + "all_results.csv"

    qa = read_qa_csv(input_data)
    count = 0
    results = []
    for index, question in enumerate(qa["question"]):
        result=[]
        myobj = {'q': question}
        # print("question is:")
        # print(myobj)
        answer = requests.post(url, json=myobj)
        # print("model return:")
        answer_dict = answer.json()
        model_res = answer_dict["response"]
        # print(model_res)
        # print("the answer from qa set:")
        qa_res = qa["answer"][index]
        # print(qa_res)
        result.append(question)
        result.append(qa_res)
        result.append(model_res)
        if model_res in qa_res or qa_res in model_res:
            count +=1
            result.insert(0,1)
        else:
            result.insert(0,0)
        # print("\n\n")

        results.append(result)
        print(result)
    results.append([count,count, count])
    print(f'correct answered: {count}')
    print("\n")
    write_qa_csv(results,output_data)