import torch
import torch.nn as nn
from model import MixModel
import pickle
import gzip
import numpy as np
from haversine import haversine
import time
import datetime
import sys
from optim import GradualWarmupScheduler


def load_obj(filename, serializer=pickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin)
    return obj


def gcn_msg(edge):
    return {'m': edge.src['h'], 'w': edge.data['w']}


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def geo_eval(y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" % (len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred])
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    mean = int(np.mean(distances))
    median = int(np.median(distances))
    acc_at_161 = acc_at_161
    return mean, median, acc_at_161


def test(model, g_m, g_r, test_mask, users, feat, classLatMedian, classLonMedian, userLocation):
    autocast = torch.cuda.amp.autocast
    model.eval()
    with torch.no_grad():
        with autocast():
            y_pred = model(g_m, g_r, feat)

        y_test_tag = torch.argmax(y_pred[test_mask], dim=1)
        y_pred_test_tag = y_test_tag.cpu().numpy()
        acc_at_161, mean, median = geo_eval(y_pred_test_tag, users[masks[2].cpu()], classLatMedian, classLonMedian, userLocation)

        return acc_at_161, mean, median


def val(model,g_m, g_r, labels, val_mask, feat):
    loss_fcn = nn.CrossEntropyLoss()
    autocast = torch.cuda.amp.autocast
    model.eval()
    with torch.no_grad():
        with autocast():
            y_pred = model(g_m, g_r, feat)
            val_loss = loss_fcn(y_pred[val_mask], labels[val_mask])
        y_test_tag = torch.argmax(y_pred[val_mask], dim=1)
        correct = (y_test_tag == labels[val_mask])
        acc = correct.sum().item() / labels[val_mask].size(0)

        return acc, val_loss


def train(model, g_m, g_r, labels, train_mask, val_mask, test_mask, U_test, feat, classLatMedian, classLonMedian, userLocation):
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    autocast = torch.cuda.amp.autocast
    best_param = None
    best_val_loss = sys.maxsize
    # MAX_LR = 3e-3
    # MIN_LR = MAX_LR / 100
    # T_max = 360
    # warmup_iter = 40
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)
    # scaler = torch.cuda.amp.GradScaler()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=MIN_LR)
    # warmup_scheduler = GradualWarmupScheduler(optimizer, warmup_iter=warmup_iter, after_scheduler=scheduler)
    # warmup_scheduler.step()  # Warm up starts from lr = 0

    start_time = time.time()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        with autocast():
            logits = model(g_m, g_r, feat)
            train_loss = loss_fcn(logits[train_mask], labels[train_mask])

        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # warmup_scheduler.step()

        train_tag = torch.argmax(logits[train_mask], dim=1)
        correct = (train_tag == labels[train_mask])
        train_acc = correct.sum().item() / labels[train_mask].size(0)
        l_train = train_loss.item()
        val_acc, val_loss = val(model,g_m, g_r, labels, val_mask, feat)
        l_val = val_loss.item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_param = model.state_dict()

        n = 10
        if epoch % n == 0:
            print("{} In Epoch {:3d} | train_loss {:.4f} | train_acc {:.4f} | val_loss {:.4f} | val_acc {:.4f}".format(
                    get_time_dif(start_time), epoch + 1, l_train, train_acc, l_val, val_acc))
    end_time = time.time()
    train_time = end_time - start_time
    model.load_state_dict(best_param)
    return best_param, train_time


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_obj("./data_dump/igr_tfidf/dump_cmu_5120.pkl")
    mention_graph, retweet_graph, users, classLatMedian, classLonMedian, userLocation = data
    men_features = mention_graph.ndata["feat"]
    labels = mention_graph.ndata["label"]
    train_mask, val_mask, test_mask = mention_graph.ndata["train_mask"], mention_graph.ndata["val_mask"], mention_graph.ndata["test_mask"]
    # 将 masks 移动到目标设备
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    # 将它们重新打包为元组
    masks = (train_mask, val_mask, test_mask)
    g_m = mention_graph.int().to(device)
    g_r = retweet_graph.int().to(device)
    feat = men_features.to(device)
    labels = labels.to(device)
    in_size = men_features.shape[1]
    result_str = []

    for i in range(3):
        model = MixModel(num_classes=129, hidden_dim=200, in_feats=in_size, dropout=0.8)
        model.to(device)  # 将模型移动到设备（GPU 或 CPU）
        best_param, train_time = train(model, g_m, g_r, labels, masks[0], masks[1], masks[2], users, feat, classLatMedian, classLonMedian, userLocation)
        mean, median, acc_at_161 = test(model, g_m, g_r, masks[2], users, feat, classLatMedian, classLonMedian, userLocation)
        r = "Mean: {:4d} | Median: {:3d} | Acc@161: {:.4f} | train_time: {:.1f}".format(mean, median, acc_at_161,train_time)
        result_str.append(r)

        print("Mean: {:4d} | Median: {:3d} | Acc@161: {:.4f} | train_time: {:.1f}".format(mean, median, acc_at_161, train_time))

    print('#' * 80)

    for st in result_str: print(st)