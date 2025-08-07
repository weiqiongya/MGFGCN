import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import csv
import re
from sklearn.cluster import KMeans
from utils import *
from collections import defaultdict, Counter
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer


class DataLoader():
    def __init__(self, path_data, bucket_size=50, encoding='ISO-8859-1',
                 celebrity_threshold=5, mindf=10, maxdf=0.2,
                 norm='l2', idf=True, btf=True, tokenizer=None, subtf=False, stops=None,
                 token_pattern=r'(?u)(?<![@])#?\b\w\w+\b', vocab=None):
        self.path_data = path_data
        self.bucket_size = bucket_size  # 区域划分bucket大小，值越大，clusters数量越少
        self.encoding = encoding
        self.celebrity_threshold = celebrity_threshold
        self.mindf = mindf
        self.maxdf = maxdf
        self.norm = norm
        self.idf = idf
        self.btf = btf
        self.tokenizer = tokenizer
        self.subtf = subtf
        self.stops = stops if stops else 'english'
        self.token_pattern = token_pattern
        self.vocab = vocab

    def read_data(self):
        console.log("[Dataset]" + self.path_data)
        train_path = os.path.join(self.path_data, 'user_info.train.gz')
        val_path = os.path.join(self.path_data, 'user_info.dev.gz')
        test_path = os.path.join(self.path_data, 'user_info.test.gz')

        train_data = pd.read_csv(train_path, delimiter='\t', encoding=self.encoding,
                                 names=['user', 'lat', 'lon', 'text'], quoting=csv.QUOTE_NONE)
        val_data = pd.read_csv(val_path, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                               quoting=csv.QUOTE_NONE)
        test_data = pd.read_csv(test_path, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                                quoting=csv.QUOTE_NONE)
        train_data.dropna(inplace=True)
        val_data.dropna(inplace=True)
        test_data.dropna(inplace=True)
        train_data['user'] = train_data['user'].apply(lambda x: str(x).lower())
        train_data.drop_duplicates(['user'], inplace=True, keep='last')
        train_data.set_index(['user'], drop=True, append=False, inplace=True)
        train_data.sort_index(inplace=True)
        val_data['user'] = val_data['user'].apply(lambda x: str(x).lower())
        val_data.drop_duplicates(['user'], inplace=True, keep='last')
        val_data.set_index(['user'], drop=True, append=False, inplace=True)
        val_data.sort_index(inplace=True)
        test_data['user'] = test_data['user'].apply(lambda x: str(x).lower())
        test_data.drop_duplicates(['user'], inplace=True, keep='last')
        test_data.set_index(['user'], drop=True, append=False, inplace=True)
        test_data.sort_index(inplace=True)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.text_train = self.train_data.text.values
        self.text_val = self.val_data.text.values
        self.text_test = self.test_data.text.values

    def build_relationship_graph(self):

        nodes = set(self.train_data.index.tolist() + self.val_data.index.tolist() + self.test_data.index.tolist())
        assert len(nodes) == len(self.train_data) + len(self.val_data) + len(self.test_data), 'duplicate target node'
        nodes_all = self.train_data.index.tolist() + self.val_data.index.tolist() + self.test_data.index.tolist()
        node_id = {node: id for id, node in enumerate(nodes_all)}

        g = nx.Graph()
        g.add_nodes_from(node_id.values())
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])

        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)

        def capture_at(data):
            for i in range(len(data)):
                user_id = node_id[data.index[i]]
                mentions = [m.lower() for m in pattern.findall(data.text[i])]
                ids = set()
                for m in mentions:
                    if m in node_id:
                        ids.add(node_id[m])
                    else:
                        _id = len(node_id)
                        node_id[m] = _id
                        ids.add(_id)
                if len(ids) > 0:
                    g.add_nodes_from(ids)
                for _id in ids:
                    g.add_edge(_id, user_id)

        capture_at(self.train_data)
        capture_at(self.val_data)
        capture_at(self.test_data)

        celebrities = []
        for i in range(len(nodes_all), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)
        console.log(
            'Removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)

        self.graph = second_order_neighbor(g, range(len(nodes_all)))

        console.log(
            'Node number: %d, Edge number: %d' % (nx.number_of_nodes(self.graph), nx.number_of_edges(self.graph)))
        return self.graph

    def build_men_relationship_graph(self):
        nodes = set(self.train_data.index.tolist() + self.val_data.index.tolist() + self.test_data.index.tolist())
        assert len(nodes) == len(self.train_data) + len(self.val_data) + len(self.test_data), 'duplicate target node'
        nodes_all = self.train_data.index.tolist() + self.val_data.index.tolist() + self.test_data.index.tolist()
        node_id = {node: id for id, node in enumerate(nodes_all)}

        g = nx.Graph()
        g.add_nodes_from(node_id.values())
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])

        pattern = r"@(\w+)\s"
        pattern = re.compile(pattern)

        def capture_at(data):
            for i in range(len(data)):
                user_id = node_id[data.index[i]]
                mentions = [m.lower() for m in pattern.findall(data.text[i])]
                ids = set()
                for m in mentions:
                    if m in node_id:
                        ids.add(node_id[m])
                    else:
                        _id = len(node_id)
                        node_id[m] = _id
                        ids.add(_id)
                if len(ids) > 0:
                    g.add_nodes_from(ids)
                for _id in ids:
                    g.add_edge(_id, user_id)

        capture_at(self.train_data)
        capture_at(self.val_data)
        capture_at(self.test_data)

        celebrities = []
        for i in range(len(nodes_all), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)
        console.log(
            'Removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)

        self.graph = second_order_neighbor(g, range(len(nodes_all)))

        console.log(
            'Node number: %d, Edge number: %d' % (nx.number_of_nodes(self.graph), nx.number_of_edges(self.graph)))
        return self.graph

    def build_ret_relationship_graph(self):
        nodes = set(self.train_data.index.tolist() + self.val_data.index.tolist() + self.test_data.index.tolist())
        assert len(nodes) == len(self.train_data) + len(self.val_data) + len(self.test_data), 'duplicate target node'
        nodes_all = self.train_data.index.tolist() + self.val_data.index.tolist() + self.test_data.index.tolist()
        node_id = {node: id for id, node in enumerate(nodes_all)}

        g = nx.Graph()
        g.add_nodes_from(node_id.values())
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])

        pattern = r"RT\s+@(\w+):"
        pattern = re.compile(pattern)

        def capture_at(data):
            for i in range(len(data)):
                user_id = node_id[data.index[i]]
                mentions = [m.lower() for m in pattern.findall(data.text[i])]
                ids = set()
                for m in mentions:
                    if m in node_id:
                        ids.add(node_id[m])
                    else:
                        _id = len(node_id)
                        node_id[m] = _id
                        ids.add(_id)
                if len(ids) > 0:
                    g.add_nodes_from(ids)
                for _id in ids:
                    g.add_edge(_id, user_id)

        capture_at(self.train_data)
        capture_at(self.val_data)
        capture_at(self.test_data)

        celebrities = []
        for i in range(len(nodes_all), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)
        console.log(
            'Removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)

        self.graph = second_order_neighbor(g, range(len(nodes_all)))

        console.log(
            'Node number: %d, Edge number: %d' % (nx.number_of_nodes(self.graph), nx.number_of_edges(self.graph)))
        return self.graph

    def build_relationship_graph_from_composed(self):
        # 1. 构建两个子图（mention 和 retweet）
        mention_graph = self.build_men_relationship_graph()
        retweet_graph = self.build_ret_relationship_graph()

        # 2. 初始化合并图（无边类型信息）
        G_total = nx.Graph()
        G_total.add_nodes_from(mention_graph.nodes())
        G_total.add_nodes_from(retweet_graph.nodes())
        G_total.add_edges_from(mention_graph.edges())
        G_total.add_edges_from(retweet_graph.edges())

        # 5. 日志输出
        console.log(
            'Node number: %d, Edge number: %d' % (nx.number_of_nodes(G_total), nx.number_of_edges(G_total)))
        return G_total

    def length(self):
        n_train = len(self.train_data)
        n_val = len(self.val_data)
        n_test = len(self.test_data)
        return n_train, n_val, n_test

    def label(self):
        # 假设 self.train_data['lat'] 和 self.train_data['lon'] 包含了训练数据的纬度和经度数据
        # 使用 K-means 对训练数据进行聚类
        kmeans = KMeans(n_clusters=129, random_state=0)
        kmeans.fit(self.train_data[['lat', 'lon']])

        # 获取训练数据的标签
        self.train_labels = kmeans.labels_

        # 使用相同的模型为验证集和测试集分配标签
        # 假设 self.val_data 和 self.test_data 也已经准备好
        self.val_labels = kmeans.predict(self.val_data[['lat', 'lon']])
        self.test_labels = kmeans.predict(self.test_data[['lat', 'lon']])

        # 存储质心
        centroids = kmeans.cluster_centers_
        self.cluster_median = {i: (centroids[i][0], centroids[i][1]) for i in range(centroids.shape[0])}

        # 返回标签
        return self.train_labels, self.val_labels, self.test_labels

    def position(self):
        train_users = self.train_data.index.tolist()
        val_users = self.val_data.index.tolist()
        test_users = self.test_data.index.tolist()
        users = np.hstack((train_users, val_users, test_users))

        train_locations = [str(a[0]) + ',' + str(a[1]) for a in self.train_data[['lat', 'lon']].values.tolist()]
        val_locations = [str(a[0]) + ',' + str(a[1]) for a in self.val_data[['lat', 'lon']].values.tolist()]
        test_locations = [str(a[0]) + ',' + str(a[1]) for a in self.test_data[['lat', 'lon']].values.tolist()]

        user_locations = {}
        for i, u in enumerate(train_users):
            user_locations[u] = train_locations[i]
        for i, u in enumerate(test_users):
            user_locations[u] = test_locations[i]
        for i, u in enumerate(val_users):
            user_locations[u] = val_locations[i]

        return user_locations, users

    def tfidf(self):
        self.vectorizer = TfidfVectorizer(max_features=150000, tokenizer=self.tokenizer,
                                          token_pattern=self.token_pattern, use_idf=self.idf,
                                          norm=self.norm, binary=self.btf, sublinear_tf=self.subtf,
                                          min_df=self.mindf, max_df=self.maxdf, ngram_range=(1, 1),
                                          stop_words=self.stops,
                                          vocabulary=self.vocab, encoding=self.encoding, dtype='float32')

        self.train_tfidf = self.vectorizer.fit_transform(self.train_data.text.values)
        self.val_tfidf = self.vectorizer.transform(self.val_data.text.values)
        self.test_tfidf = self.vectorizer.transform(self.test_data.text.values)
        console.log("training    n_samples: %d, n_features: %d" % self.train_tfidf.shape)
        console.log("development n_samples: %d, n_features: %d" % self.val_tfidf.shape)
        console.log("test        n_samples: %d, n_features: %d" % self.test_tfidf.shape)


    def tfidf_IGR(self, train_classes, igr_threshold=0.0):  # 可调节阈值

        # 第一次生成原始 TF-IDF 特征
        vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=self.token_pattern, use_idf=self.idf,
                                     norm=self.norm, binary=self.btf, sublinear_tf=self.subtf,
                                     min_df=self.mindf, max_df=self.maxdf, ngram_range=(1, 1),
                                     stop_words=self.stops,
                                     vocabulary=self.vocab, encoding=self.encoding, dtype='float32')

        import warnings
        warnings.filterwarnings("ignore", message="Clustering metrics expects discrete values*", category=UserWarning)

        train_tfidf = vectorizer.fit_transform(self.train_data.text.values)
        feature_names = vectorizer.get_feature_names_out()
        print("原始vocabulary数量:", len(feature_names))

        # 计算 IGR (信息增益率)
        mi = mutual_info_classif(train_tfidf, train_classes, discrete_features=True)  # MI for each feature
        entropy_y = -np.sum(
            [np.mean(train_classes == c) * np.log2(np.mean(train_classes == c)) for c in np.unique(train_classes)]
        )
        igr = mi / entropy_y  # 信息增益率

        # 根据阈值筛选特征
        selected_indices = np.where(igr >= igr_threshold)[0]
        selected_features = feature_names[selected_indices]
        print("筛选后vocabulary数量:", len(selected_features))

        # 更新 vocabulary
        self.new_vocab = {word: idx for idx, word in enumerate(selected_features)}

        # 第二次使用新的vocab生成 TF-IDF 特征
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=self.token_pattern, use_idf=self.idf,
                                          norm=self.norm, binary=self.btf, sublinear_tf=self.subtf,
                                          min_df=self.mindf, max_df=self.maxdf, ngram_range=(1, 1),
                                          stop_words=self.stops,
                                          vocabulary=self.new_vocab, encoding=self.encoding, dtype='float32')

        self.train_tfidf = self.vectorizer.fit_transform(self.train_data.text.values)
        self.val_tfidf = self.vectorizer.transform(self.val_data.text.values)
        self.test_tfidf = self.vectorizer.transform(self.test_data.text.values)

        print("training    n_samples: %d, n_features: %d" % self.train_tfidf.shape)
        print("development n_samples: %d, n_features: %d" % self.val_tfidf.shape)
        print("test        n_samples: %d, n_features: %d" % self.test_tfidf.shape)


    def igr_tfidf(self, train_classes, top_k=0):
        """
        先根据训练文本和标签计算 IGR（信息增益率），
        然后选取 top_k 个最重要的词，生成 TF-IDF 特征。
        """
        if self.tokenizer is None:
            self.tokenizer = lambda x: x.split()

        texts = self.train_data.text.values
        labels = np.array(train_classes)
        label_classes = np.unique(labels)
        N = len(texts)
        label_counts = Counter(labels)

        # ---------- 计算每个词的 IGR ----------
        word_doc_label = defaultdict(lambda: Counter())  # word -> label -> count
        word_doc_freq = Counter()

        for text, label in zip(texts, labels):
            words = set(self.tokenizer(text))  # 去重后再统计是否出现
            for word in words:
                word_doc_label[word][label] += 1
                word_doc_freq[word] += 1

        # 标签总熵 H(Y)
        total_entropy = -sum(
            (label_counts[l] / N) * math.log2(label_counts[l] / N) for l in label_classes
        )

        igr_scores = {}
        for word, label_freqs in word_doc_label.items():
            Pw = word_doc_freq[word] / N
            Pnw = 1 - Pw

            # P(Y|w)
            cond_probs_w = [(label_freqs[l] / word_doc_freq[word]) if word_doc_freq[word] > 0 else 0 for l in
                            label_classes]
            entropy_given_w = -sum(p * math.log2(p) for p in cond_probs_w if p > 0)

            # P(Y|~w)
            label_freqs_nw = {l: label_counts[l] - label_freqs[l] for l in label_classes}
            nw_total = N - word_doc_freq[word]
            cond_probs_nw = [(label_freqs_nw[l] / nw_total) if nw_total > 0 else 0 for l in label_classes]
            entropy_given_nw = -sum(p * math.log2(p) for p in cond_probs_nw if p > 0)

            # 信息增益率
            info_gain = total_entropy - (Pw * entropy_given_w + Pnw * entropy_given_nw)
            igr = info_gain / total_entropy if total_entropy > 0 else 0
            igr_scores[word] = igr

        # ---------- 筛选前 top_k 个词 ----------
        sorted_words = sorted(igr_scores.items(), key=lambda x: x[1], reverse=True)
        print(len(sorted_words))
        selected_vocab = [word for word, _ in sorted_words[:top_k]]
        print("IGR 选出的特征数:", len(selected_vocab))

        # ---------- 用筛选后的词表做 TF-IDF ----------
        self.new_vocab = {word: idx for idx, word in enumerate(selected_vocab)}

        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=self.token_pattern,
                                          use_idf=self.idf, norm=self.norm, binary=self.btf,
                                          sublinear_tf=self.subtf, min_df=self.mindf, max_df=self.maxdf,
                                          ngram_range=(1, 1), stop_words=self.stops,
                                          vocabulary=self.new_vocab, encoding=self.encoding, dtype='float32')

        self.train_tfidf = self.vectorizer.fit_transform(self.train_data.text.values)
        self.val_tfidf = self.vectorizer.transform(self.val_data.text.values)
        self.test_tfidf = self.vectorizer.transform(self.test_data.text.values)

        print("training    n_samples: %d, n_features: %d" % self.train_tfidf.shape)
        print("development n_samples: %d, n_features: %d" % self.val_tfidf.shape)
        print("test        n_samples: %d, n_features: %d" % self.test_tfidf.shape)


if __name__ == '__main__':

    dl = DataLoader("data/cmu")
    dl.read_data()
    # dl.build_relationship_graph()
    # dl.build_relationship_graph_from_composed()
    # # # dl.build_men_relationship_graph()
    dl.tfidf()
    # train_classes, val_classes, test_classes = dl.label()
    # # print(train_classes)
    # # print(train_classes.dtype)
    # # dl.tfidf_IGR(train_classes, igr_threshold=0.0014)
    # dl.igr_tfidf(train_classes, top_k=5120)


