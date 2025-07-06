import json

with open('alpaca_data.json', 'r') as f:
    data = json.load(f)

def split_data(data, ratio):
    n = int(len(data) * ratio)
    return data[:n], data[n:]
                
trainset, testset = split_data(data, 0.8)
cleanset, noisyset = split_data(trainset, 0.5)
testset, valset = split_data(testset, 0.5)

for sample in noisyset:
    sample['output'] = ''

with open("train_clean.json", "w") as f:
    json.dump(cleanset, f)

with open("train_noisy.json", "w") as f:
    json.dump(noisyset, f)

with open("val.json", "w") as f:
    json.dump(valset, f)

with open("test.json", "w") as f:
    json.dump(testset, f)