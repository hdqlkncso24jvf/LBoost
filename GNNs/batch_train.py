from models import *
from dataset import *
from sklearn.metrics import f1_score
import torch
from args import *
import os
import torch.nn.functional as F

arg = get_parameter()

# print(arg)

dataset = Dataset(arg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if arg.soft:
    soft_criterion = torch.nn.KLDivLoss(reduction='batchmean')
criterion = torch.nn.CrossEntropyLoss()
ground_truth = torch.tensor(dataset.labels).to(device).long()
if arg.soft:
    soft_labels = dataset.soft_labels.to(device).long()
train_mask = dataset.train_mask
test_mask = dataset.test_mask
valid_mask = dataset.valid_mask

x_features = dataset.x.to(device)
edge_index = torch.tensor(dataset.edge_index).to(device).long()

for gnn_type in ["GCN", "GAT", "GraphSAGE"]:
    if gnn_type == "GCN":
        model = GCN(dataset.embed_size, arg.hidden_dim, dataset.class_num).to(device)
    elif gnn_type == "GAT":
        model = GAT(dataset.embed_size, arg.hidden_dim, dataset.class_num).to(device)
    else:
        model = GraphSAGE(dataset.embed_size, arg.hidden_dim, dataset.class_num).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    model.train()

    pre_val_f1 = 0

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~Training GNN~~~~~~~~~~~~~~~~~~~~~~~~~~")

    for epoch in range(arg.epoch):
        pred = model(x_features, edge_index)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if arg.soft:
            loss = soft_criterion(pred[train_mask].float(), soft_labels[train_mask].float())
        else:
            loss = criterion(pred[train_mask], ground_truth[train_mask])


        train_output = pred[train_mask].max(1)[1].cpu().numpy()
        train_true_labels = ground_truth[train_mask].cpu().numpy()
        train_micro_f1 = f1_score(train_true_labels, train_output, average='micro')
        print(
            f'Train: Micro-F1 Score of {gnn_type} with {arg.noise_ratio} noise ratio on {gnn_type}: {train_micro_f1}')

        # val_output = pred[valid_mask].max(1)[1].cpu().numpy()
        # val_true_labels = ground_truth[valid_mask].cpu().numpy()
        # val_micro_f1 = f1_score(val_true_labels, val_output, average='micro')

        # if val_micro_f1 < pre_val_f1 and arg.epoch > 500:
        #     break

        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~Saving GNN~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print("Saving the model")
    directory = "models/{}/".format(gnn_type)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model, directory + "{}_{}_noise_rate_{}.ckp".format(gnn_type, arg.dataset, arg.noise_ratio))
