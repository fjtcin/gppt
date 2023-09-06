import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
import torch, gc

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type='gcn'):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregator_type))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator_type))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggregator_type))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, aggregator_type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    def get_e(self):
        return self.embedding_x
    def forward(self, blocks, x):
        h = self.dropout(x)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        self.embedding_x=h
        return h

    def forward_smc(self, g, x):
        h = h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        self.embedding_x=h
        return h

    def inference(self, g, x, device, batch_size):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes, device=device)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False)


            for input_nodes, output_nodes, blocks in dataloader:#tqdm.tqdm(dataloader):
                block = blocks[0]
                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[output_nodes] = h
                #gc.collect()
                #torch.cuda.empty_cache()

            x = y
        return y

def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.numpy(force=True)
    labels = labels.numpy(force=True)
    train_nids = train_nids.numpy(force=True)
    train_labels = labels[train_nids]
    val_nids = val_nids.numpy(force=True)
    val_labels = labels[val_nids]
    test_nids = test_nids.numpy(force=True)
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test
