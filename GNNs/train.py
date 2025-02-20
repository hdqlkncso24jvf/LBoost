from models import *
from dataset import *
from sklearn.metrics import f1_score
import torch
from args import *
import os
import time
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


arg = get_parameter()

if arg.soft:
    soft_criterion = torch.nn.KLDivLoss(reduction='batchmean')

criterion = torch.nn.CrossEntropyLoss()

dataset = Dataset(arg)

assert arg.model_type in ["GCN", "GAT", "GraphSAGE"]

if arg.model_type == "GCN":
    model = GCN(dataset.embed_size, arg.feature * arg.hidden_dim, dataset.class_num).to(device)
elif arg.model_type == "GAT":
    model = GAT(dataset.embed_size, arg.feature * arg.hidden_dim, dataset.class_num).to(device)
else:
    model = GraphSAGE(dataset.embed_size, arg.feature * arg.hidden_dim, dataset.class_num).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
model.train()

ground_truth = torch.tensor(dataset.labels).to(device).long()
if arg.soft:
    soft_labels = dataset.soft_labels.to(device).long()

train_mask = dataset.train_mask
test_mask = dataset.test_mask
valid_mask = dataset.valid_mask

x_features = dataset.x.to(device)
edge_index = torch.tensor(dataset.edge_index).to(device).long()

pre_val_f1 = 0

print("~~~~~~~~~~~~~~~~~~~~~~~~~~Training GNN~~~~~~~~~~~~~~~~~~~~~~~~~~")
start_time = time.time()

for epoch in range(arg.epoch):
    pred = model(x_features, edge_index)
    if arg.soft:
        loss = soft_criterion(pred[train_mask].float(), soft_labels[train_mask].float())
    else:
        loss = criterion(pred[train_mask], ground_truth[train_mask])

    if (1 + epoch) % 100 == 0:
        train_output = pred[train_mask].max(1)[1].cpu().numpy()
        train_true_labels = ground_truth[train_mask].cpu().numpy()
        train_micro_f1 = f1_score(train_true_labels, train_output, average='micro')
        print(
            f'Test: Micro-F1 Score of {arg.model_type} with {arg.noise_ratio} noise ratio on {arg.model_type}: {train_micro_f1}')

        val_output = pred[valid_mask].max(1)[1].cpu().numpy()
        val_true_labels = ground_truth[valid_mask].cpu().numpy()
        val_micro_f1 = f1_score(val_true_labels, val_output, average='micro')
        print(f'Micro-F1 Score in valid set: {val_micro_f1}')
        if val_micro_f1 < pre_val_f1 and arg.epoch > 100:
            break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end_time = time.time()

print(f"Traing time of {arg.dataset} with {arg.feature} features：{end_time - start_time} seconds")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~Saving GNN~~~~~~~~~~~~~~~~~~~~~~~~~~~")

print("Saving the model")
directory = "models/{}/".format(arg.model_type)
if not os.path.exists(directory):
    os.makedirs(directory)
torch.save(model, directory + "{}_{}_noise_rate_{}.ckp".format(arg.model_type, arg.dataset, arg.noise_ratio))
