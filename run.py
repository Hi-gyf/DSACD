import torch
import numpy as np
import random
import torch.optim as optim
import torch.nn as nn
from argparse import ArgumentParser
from data_loader import ScoreDataLoader
from modules import SACD
from utils import evaluate

parser = ArgumentParser("DSACD")
# runtime args
parser.add_argument("--dataset", type=str, help='pisa2015 / junyi / ednet / assistment2017', default="assistment2017")
parser.add_argument("--device", type=str, help='cpu or gpu', default="cpu")
parser.add_argument("--train_rate", type=float, help='train rate', default=0.8)
parser.add_argument("--batch_size", type=int, help='batch size', default=16)
parser.add_argument("--lr", type=float, help='learning rate', default=0.002)
parser.add_argument("--epoch", type=int, help='cpu or gpu', default=5)
parser.add_argument("--K", type=int, help='K fold', default=5)
parser.add_argument("--hid1", type=int, help='hidden 1', default=512)
parser.add_argument("--hid2", type=int, help='hidden 2', default=256)
parser.add_argument("--step", type=int, help='response process steps', default=10)
args = parser.parse_args()

device = args.device
if args.device != 'cpu':
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

dataset_file = args.dataset
train_rate = args.train_rate
batch_size = args.batch_size
lr = args.lr
epoch = args.epoch
K = args.K
hid1, hid2 = args.hid1, args.hid2
step = args.step
print(str(args))

dataloader = ScoreDataLoader(dataset_file=dataset_file, batch_size=batch_size, train_rate=train_rate, K=K)
train_val_dataloader = dataloader.get_train_val_dataloader()
assert K == len(train_val_dataloader)

n_stu, n_pro, n_know = dataloader.get_data_shape()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(123)

# Train
t_acc, t_auc, t_precision, t_recall, t_f1, t_rmse = 0, 0, 0, 0, 0, 0
for k in range(K):
    print('Cross ', k + 1, ' of ', K)
    model = SACD(n_stu, n_pro, n_know, hid1, hid2, step)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.NLLLoss()

    train_dataloader = train_val_dataloader[k]['train']
    val_dataloader = train_val_dataloader[k]['val']
    test_dataloader = dataloader.get_test_dataloader()

    running_loss = 0.0
    for e in range(epoch):
        model.train()
        for index, (sid, pid, Q, rt, r) in enumerate(train_dataloader):
            sid, pid, Q, rt, r = sid.to(device), pid.to(device), Q.to(device), rt.to(device), r.to(device)
            optimizer.zero_grad()
            output_1= model.forward(sid, pid, Q, rt, device)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output), r.long())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.apply_clipper()

            running_loss += loss.item()
            if index % 1000 == 999:
                print('[%d, %2d, %5d] loss: %.3f' % (k + 1, e + 1, index + 1, running_loss / 1000))
                running_loss = 0.0

        model.eval()
        e_pred, e_label = np.array([]), np.array([])
        for index, (sid, pid, Q, rt, r) in enumerate(val_dataloader):
            sid, pid, Q, rt, r = sid.to(device), pid.to(device), Q.to(device), rt.to(device), r.to(device)
            val_output = model.forward(sid, pid, Q, rt, device)
            val_output, r = val_output.to(torch.device('cpu')).detach().numpy().flatten(), r.to(torch.device('cpu')).numpy()
            e_pred = np.concatenate((e_pred, val_output))
            e_label = np.concatenate((e_label, r))
        print('-------------Val: epoch ' + str(e + 1) + '-----------------')
        evaluate(e_pred, e_label)

    model.eval()
    c_pred, c_label = np.array([]), np.array([])
    for index, (sid, pid, Q, rt, r) in enumerate(test_dataloader):
        sid, pid, Q, rt, r = sid.to(device), pid.to(device), Q.to(device), rt.to(device), r.to(device)
        val_output = model.forward(sid, pid, Q, rt, device, test=True)
        val_output, r = val_output.to(torch.device('cpu')).detach().numpy().flatten(), r.to(torch.device('cpu')).numpy()
        c_pred = np.concatenate((c_pred, val_output))
        c_label = np.concatenate((c_label, r))
    print('========Test: Cross ' + str(k + 1) + '===============')
    acc, auc, precision, recall, f1, rmse = evaluate(c_pred, c_label)
    t_acc += acc
    t_auc += auc
    t_precision += precision
    t_recall += recall
    t_f1 += f1
    t_rmse += rmse
print('final acc: {}, final auc: {}, final precision: {}, final recall: {}, final f1: {}, rmse: {}\n'.format(t_acc / K, t_auc / K,
                                                                                                 t_precision / K,
                                                                                                 t_recall / K, t_f1 / K, t_rmse / K))
with open('./res/' + dataset_file + '.txt', 'a', encoding='utf8') as f:
    f.write(str(args))
    f.write('\nfinal acc: {}, final auc: {}, final precision: {}, final recall: {}, final f1: {}, rmse: {}\n'.format(t_acc / K, t_auc / K,
                                                                                                 t_precision / K,
                                                                                                 t_recall / K, t_f1 / K, t_rmse / K))
    f.write('----------------------------------------------------------------------------------------\n')

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()
save_snapshot(model, './res/model.pkl')
