import dgl
from dgl.data import DGLDataset
import networkx as nx
import scipy as sp
from utils import *
from network_data_processing import *


class CmuDataset(DGLDataset):
    def __init__(self, transform=None):
        super(CmuDataset, self).__init__(name='na', transform=transform)

    def process(self):
        dl = DataLoader("data/na")
        dl.read_data()
        train_classes, val_classes, test_classes = dl.label()

        dl.igr_tfidf(train_classes, top_k=20000)
        X = sp.sparse.vstack([dl.train_tfidf, dl.val_tfidf, dl.test_tfidf])
        X_dense = X.todense()
        feat = torch.from_numpy(X_dense).float()

        classes = np.concatenate((train_classes, val_classes, test_classes), axis=0)

        m_g = dl.build_men_relationship_graph().to_directed()
        r_g = dl.build_ret_relationship_graph().to_directed()
        # g = dl.build_relationship_graph_from_composed().to_directed()

        self.mention_graph = dgl.from_networkx(m_g)
        self.retweet_graph = dgl.from_networkx(r_g)
        # self.graph = dgl.from_networkx(g)


        self.mention_graph.ndata['feat'] = feat
        self.retweet_graph.ndata['feat'] = feat
        # self.graph.ndata['feat'] = feat

        self.mention_graph.ndata['label'] = torch.LongTensor(classes)
        self.retweet_graph.ndata['label'] = torch.LongTensor(classes)
        # self.graph.ndata['label'] = torch.LongTensor(classes)

        # Mask
        n_train, n_val, n_test = dl.length()
        n_nodes = n_train + n_val + n_test  # 5685 1895 1895
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.mention_graph.ndata['train_mask'] = train_mask
        self.mention_graph.ndata['val_mask'] = val_mask
        self.mention_graph.ndata['test_mask'] = test_mask
        self.retweet_graph.ndata['train_mask'] = train_mask
        self.retweet_graph.ndata['val_mask'] = val_mask
        self.retweet_graph.ndata['test_mask'] = test_mask
        # self.graph.ndata['train_mask'] = train_mask
        # self.graph.ndata['val_mask'] = val_mask
        # self.graph.ndata['test_mask'] = test_mask

        self.mention_graph = dgl.add_self_loop(self.mention_graph)
        self.retweet_graph = dgl.add_self_loop(self.retweet_graph)
        # self.graph = dgl.add_self_loop(self.graph)

        out_size = torch.max(self.mention_graph.ndata["label"]).item() + 1
        print("out_size: ", out_size)

        userLocation, users = dl.position()
        classLatMedian = {str(c): dl.cluster_median[c][0] for c in dl.cluster_median}
        classLonMedian = {str(c): dl.cluster_median[c][1] for c in dl.cluster_median}
        data = (self.mention_graph, self.retweet_graph, users, classLatMedian, classLonMedian, userLocation)
        # print(self.graph)
        dump_folder = './data_dump/igr_tfidf/'
        dumpfile = dump_folder + 'dump_na_20000.pkl'
        console.log('dumping data in {} ...'.format(dumpfile))
        dump_obj(data, dumpfile)
        console.log('data dump finished!')

    def __getitem__(self, i):
        return self.mention_graph

    def __len__(self):
        return 1


if __name__ == '__main__':
    dump_folder = './data_dump/'
    _dump_base_data = True
    max_len = 93

    if _dump_base_data:
        CmuDataset()