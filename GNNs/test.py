from dataset import *
from sklearn.metrics import f1_score
import torch
from args import *
import os

arg = get_parameter()

dataset = Dataset(arg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

directory = "models/{}/".format(arg.model_type)
model_path = "{}_{}_noise_rate_{}.ckp".format(arg.model_type, arg.dataset, arg.noise_ratio)
model = torch.load(directory + model_path, map_location=device)

ground_truth = torch.tensor(dataset.labels).to(device).long()
train_mask = dataset.train_mask
test_mask = dataset.test_mask
valid_mask = dataset.valid_mask

x_features = dataset.x.to(device)
edge_index = torch.tensor(dataset.edge_index).to(device).long()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~Testing  GNN~~~~~~~~~~~~~~~~~~~~~~~~~~")

model.eval()
pred = model(x_features, edge_index)

output = pred[test_mask].max(1)[1].cpu().numpy()
true_labels = ground_truth[test_mask].cpu().numpy()
micro_f1 = f1_score(true_labels, output, average='micro')

print(f'Micro-F1 Score of {arg.model_type} with {arg.noise_ratio} noise ratio on {arg.dataset}: {micro_f1}')
