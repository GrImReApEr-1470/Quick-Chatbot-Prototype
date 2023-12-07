import pandas as pd
import cleantext

# Load the CSV file into a DataFrame
csv_path = 'data/topical_chat.csv'
df = pd.read_csv(csv_path)
convo1 = list(df.message)
print(convo1[-10:])

convo2 = []
with open(r'data/human_chat.txt', 'r', encoding='utf-8') as w:
    for line in w:
        try:
            convo2.append(line.split(': ')[1].split("\n")[0])
        except:
            print('done')
print(convo2[-10:])

processed_text = []

def clean(textdata):
    ind = 0
    for i in textdata:
        if ind%50==0:
            print(ind/len(textdata)*100, '% done')
        ind+=1

        processed_text.append(cleantext.clean(str(i), extra_spaces=True, lowercase=False, stopwords=False, stemming=False, numbers=False, punct=False, clean_all = False))
        # print(i, processed_text[-1])
    
    return processed_text[-10:]

    

clean(convo1)
print(processed_text[-10:])
clean(convo2)
print(processed_text[-10:])

questions = processed_text[:-1]
answers = processed_text[1:]

dataset_file = "dataset.txt"

with open(dataset_file, "w", encoding="utf-8") as file:
    for query, reply in zip(questions, answers):
        file.write(f"{query}\t{reply}\n")






with open('dataset.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

processed_lines = []

for line in lines:
    try:
        query, reply = line.strip().split('\t')
        processed_lines.append(f"{query} <|endoftext|> {reply} <|endoftext|>\n")
    except:
        continue

with open('processed_dataset.txt', 'w', encoding='utf-8') as file:
    file.writelines(processed_lines)
