from dataset import *
import torch
from args import *
from torch import nn
import random
import copy

arg = get_parameter()
dataset = Dataset(arg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

directory = "models/{}/".format(arg.model_type)
model_path = "{}_{}_noise_rate_{}.ckp".format(arg.model_type, arg.dataset, arg.noise_ratio)
model = torch.load(directory + model_path, map_location=device)

x_features = dataset.x.to(device)
edge_index = torch.tensor(dataset.edge_index).to(device).long()

model.eval()
target_classes = list(range(dataset.class_num))
label_dict = dataset.label_id2txt

error_rates = [5, 10, 15, 20]

id2class = {}
num2id = []

label_set = set()

with open("data/{}/label_with_noise_0.txt".format(arg.dataset)) as f:
    for line in f.readlines():
        line = line[:-1:].split('\t')
        label_set.add(line[-1])
        num2id.append(line[0])
        id2class[line[0]] = line[-1]

total_cnt = len(num2id)
label_set = list(label_set)
katsudan = int(error_rates[-1] * total_cnt / 100)
selected_elements = random.sample(list(range(0, total_cnt)), katsudan)

for error_rate in error_rates:
    noisy_id2class = copy.deepcopy(id2class)
    cleaned_id2class = copy.deepcopy(id2class)

    katsudan = int(error_rate * total_cnt / 100)
    noise_ids = selected_elements[:katsudan:]
    output = model(x_features, edge_index).to(device)
    loss_fn = nn.CrossEntropyLoss()

    for noise_id in noise_ids:
        node_id = num2id[noise_id]
        original_class = id2class[node_id]
        perturb_class = random.choice(label_set)
        max_gradients = 0
        for target_class in target_classes:
            loss = loss_fn(output[noise_id], torch.tensor(target_class).to(device))
            model.zero_grad()
            loss.backward(retain_graph=True)
            l2_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    l2_norm += torch.norm(param.grad).item() ** 2
            l2_norm = l2_norm ** 0.5
            if l2_norm > max_gradients:
                max_gradients = l2_norm
                perturb_class = label_dict[target_class]

        noisy_id2class[node_id] = perturb_class

    with open("label_with_noise_{}.txt".format(error_rate), 'w') as f:
        for key in noisy_id2class:
            f.write(f"{key}\t{noisy_id2class[key]}\n")
