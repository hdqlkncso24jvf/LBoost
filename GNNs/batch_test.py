from sklearn.metrics import f1_score
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arg = get_parameter()

# print(arg)

dataset = Dataset(arg)
ground_truth = torch.tensor(dataset.labels).to(device).long()
train_mask = dataset.train_mask
test_mask = dataset.test_mask
valid_mask = dataset.valid_mask

x_features = dataset.x.to(device)
edge_index = torch.tensor(dataset.edge_index).to(device).long()
print("~~~~~~~~~~~~~~~~~~~~~~~~~~Testing  GNN~~~~~~~~~~~~~~~~~~~~~~~~~~")

for gnn_type in ["GCN", "GraphSAGE", "GAT"]:
    directory = "models/{}/".format(gnn_type)
    model_path = "{}_{}_noise_rate_0.ckp".format(gnn_type, arg.dataset, arg.noise_ratio)
    model = torch.load(directory + model_path, map_location=device)

    model.eval()
    pred = model(x_features, edge_index)

    output = pred[test_mask].max(1)[1].cpu().numpy()
    true_labels = ground_truth[test_mask].cpu().numpy()
    print(output)
    print(true_labels)
    micro_f1 = f1_score(true_labels, output, average='micro')

    print(f'Micro-F1 Score of {gnn_type} with {arg.noise_ratio} noise ratio on {arg.dataset}: {micro_f1}')
