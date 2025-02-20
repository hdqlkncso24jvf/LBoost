from dataset import *
import torch
from args import *
from torch import nn
import torch.nn.functional as F

arg = get_parameter()

with open("data/{}/label2id.json".format(arg.dataset)) as f:
    label_txt2id = json.load(f)

num_classes = len(label_txt2id)
raw_node2raw_id = {}
raw_id2valid_id = {}

dataset = Dataset(arg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

directory = "models/{}/".format(arg.model_type)
model_path = "{}_{}_noise_rate_{}.ckp".format(arg.model_type, arg.dataset, arg.noise_ratio)
model = torch.load(directory + model_path, map_location=device)

x_features = dataset.x.to(device)
edge_index = torch.tensor(dataset.edge_index).to(device).long()

model.eval()

# Fill this list with node that had been assigned multi-labels in LBR Chase procedure
multi_labels_nodes = {
    0: [7, 21],
    124: [5, 9],
    345: [4, 8, 11, 16]
}

output = model(x_features, edge_index).to(device)
loss_fn = nn.CrossEntropyLoss()


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes, dtype=np.float32)[labels]


def modify_one_hot(multi_label_node):
    modified_vector = torch.zeros(dataset.class_num)
    for target_class in multi_labels_nodes[multi_label_node]:
        loss = loss_fn(output[multi_label_node], torch.tensor(target_class).to(device))
        model.zero_grad()
        loss.backward(retain_graph=True)
        l2_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                l2_norm += torch.norm(param.grad).item() ** 2
        l2_norm = l2_norm ** 0.5
        modified_vector[target_class] = l2_norm

    modified_vector = F.softmax(modified_vector, dim=0)
    return modified_vector


id2embeddings = torch.load("data/{}/features_1.pth".format(arg.dataset))
ground_truth = []

id2label = {}
with open("data/{}/node.txt".format(arg.dataset)) as f:
    for line in f.readlines():
        line = line[:-1:].split('\t')
        raw_node2raw_id[line[1]] = line[-1]

with open("data/{}/label_with_noise_0.txt".format(arg.dataset)) as f:
    for line in f.readlines():
        line = line[:-1:].split('\t')
        id2label[raw_node2raw_id[line[0]]] = label_txt2id[line[-1]]

for raw_id in id2embeddings:
    if raw_id in id2label and raw_id in id2embeddings:
        ground_truth.append(id2label[raw_id])

encoded_labels = one_hot_encode(ground_truth, num_classes)
modified_encoded_labels = []
for i in range(len(encoded_labels)):
    if i in multi_labels_nodes:
        modified_encoded_labels.append(modify_one_hot(i))
    else:
        modified_encoded_labels.append(encoded_labels[i])
encoded_labels_tensor = torch.tensor(np.array(modified_encoded_labels), dtype=torch.float32)
torch.save(encoded_labels_tensor, 'softlabels.pth')
