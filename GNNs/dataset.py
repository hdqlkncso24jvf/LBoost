import collections
import torch
import numpy as np
from args import *
import json


class Dataset:
    def __init__(self, arg):

        self.arg = arg

        self.data_path = "data/{}".format(arg.dataset)

        self.x = np.array([])
        self.edge_index = [[], []]
        self.train_mask = None
        self.test_mask = None
        self.valid_mask = None
        self.labels = np.array([])
        self.label_txt2id = {}
        self.label_id2txt = {}
        self.class_num = 0
        self.embed_size = arg.feature * 64

        self.load_data()
        self.split_data()

    def load_data(self):
        raw_node2raw_id = {}
        raw_id2valid_id = {}

        raw_nodes = set()

        in_nodes = collections.defaultdict(set)
        out_nodes = collections.defaultdict(set)

        id2embeddings = torch.load(self.data_path + '/features_{}.pth'.format(self.arg.feature))

        id2label = {}

        with open(self.data_path + "/" + "label2id.json") as f:
            self.label_txt2id = json.load(f)

        for key in self.label_txt2id:
            self.label_id2txt[self.label_txt2id[key]] = key
        self.class_num = len(self.label_txt2id)

        with open(self.data_path + "/" + "node.txt") as f:
            for line in f.readlines():
                line = line[:-1:].split('\t')
                raw_node2raw_id[line[1]] = line[-1]

        if self.arg.cleaned:
            with open(self.data_path + "/" + "label_with_noise_{}_cleaned.txt".format(self.arg.noise_ratio)) as f:
                for line in f.readlines():
                    line = line[:-1:].split('\t')
                    id2label[raw_node2raw_id[line[0]]] = self.label_txt2id[line[-1]]
        else:
            with open(self.data_path + "/" + "label_with_noise_{}.txt".format(self.arg.noise_ratio)) as f:
                for line in f.readlines():
                    line = line[:-1:].split('\t')
                    id2label[raw_node2raw_id[line[0]]] = self.label_txt2id[line[-1]]

        with open(self.data_path + "/" + "edge.txt") as f:
            for line in f.readlines():
                line = line[:-1:].split('\t')
                in_nodes[line[1]].add(line[0])
                out_nodes[line[0]].add(line[1])
                if line[0] not in id2embeddings:
                    raw_nodes.add(line[0])
                if line[1] not in id2embeddings:
                    raw_nodes.add(line[1])

        def prune_graph(raw_nodes, in_nodes, out_nodes):
            # 处理每个待删除节点
            while raw_nodes:
                current = raw_nodes.pop()

                # 获取当前节点的所有入边和出边节点
                current_in_nodes = in_nodes.pop(current, set())
                current_out_nodes = out_nodes.pop(current, set())

                # 更新入边节点的出边信息
                for in_node in current_in_nodes:
                    if in_node in out_nodes:  # 确保in_node未被删除
                        out_nodes[in_node].discard(current)  # 移除指向当前节点的边
                        out_nodes[in_node].update(current_out_nodes)  # 添加新的出边

                # 更新出边节点的入边信息
                for out_node in current_out_nodes:
                    if out_node in in_nodes:  # 确保out_node未被删除
                        in_nodes[out_node].discard(current)  # 移除来自当前节点的边
                        in_nodes[out_node].update(current_in_nodes)  # 添加新的入边

            edges = []

            for node in in_nodes:
                for in_node in in_nodes[node]:
                    if node != in_node:
                        edges.append((node, in_node))

            return edges

        edges = prune_graph(raw_nodes, in_nodes, out_nodes)
        x_features = []
        ground_truth = []

        for raw_id in id2embeddings:
            raw_id2valid_id[raw_id] = len(raw_id2valid_id)
            if raw_id in id2label and raw_id in id2embeddings:
                x_features.append(id2embeddings[raw_id])
                ground_truth.append(id2label[raw_id])

        self.x = torch.stack(x_features, dim=0)

        if self.arg.soft:
            self.soft_labels = torch.load(self.data_path + '/softlabels.pth')
        self.labels = np.array(ground_truth)

        for edge in edges:
            src, tgt = edge
            self.edge_index[0].append(raw_id2valid_id[src])
            self.edge_index[1].append(raw_id2valid_id[tgt])

    def split_data(self):
        seed = 42
        torch.manual_seed(seed)
        train_ratio = 0.25
        test_ratio = 0.5
        valid_ratio = 0.25

        assert train_ratio + test_ratio + valid_ratio == 1

        # 获取数据总量
        total_count = self.x.shape[0]

        # 计算训练集、测试集和验证集的大小
        train_size = int(total_count * train_ratio)
        test_size = int(total_count * test_ratio)

        # 打乱数据
        indices = torch.randperm(total_count)

        # 切分数据
        train_indices = indices[:train_size]
        test_indices = indices[train_size:train_size + test_size]
        val_indices = indices[train_size + test_size:]

        # 根据索引获取数据子集
        self.train_mask = train_indices
        self.test_mask = test_indices
        self.valid_mask = val_indices
