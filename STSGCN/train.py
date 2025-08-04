import os
import time
import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
import tqdm

from engine import trainer
from utils import *
from model import STSGCN
import ast

DATASET = 'PEMSD4'

config_file = './{}.conf'.format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)


parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--no_cuda', default=True, action="store_true", help="没有GPU")
parser.add_argument('--data', type=str, default=config['data']['data'], help='data path')
parser.add_argument('--sensors_distance', type=str, default=config['data']['sensors_distance'], help='节点距离文件')
parser.add_argument('--column_wise', type=eval, default=config['data']['column_wise'],
                    help='是指列元素的级别上进行归一，否则是全样本取值')
parser.add_argument('--normalizer', type=str, default=config['data']['normalizer'], help='归一化方式')
parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'], help="batch大小")

parser.add_argument('--num_of_vertices', type=int, default=config['model']['num_of_vertices'], help='传感器数量')
parser.add_argument('--construct_type', type=str, default=config['model']['construct_type'],
                    help="构图方式  {connectivity, distance}")
parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'], help='输入维度')
parser.add_argument('--hidden_dims', type=list, default=ast.literal_eval(config['model']['hidden_dims']),
                    help='中间各STSGCL层的卷积操作维度')
parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'],
                    help='第一层输入层的维度')
parser.add_argument('--out_layer_dim', type=int, default=config['model']['out_layer_dim'], help='输出模块中间层维度')
parser.add_argument("--history", type=int, default=config['model']['history'], help="每个样本输入的离散时序")
parser.add_argument("--horizon", type=int, default=config['model']['horizon'], help="每个样本输出的离散时序")
parser.add_argument("--strides", type=int, default=config['model']['strides'], help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3")
parser.add_argument("--temporal_emb", type=eval, default=config['model']['temporal_emb'], help="是否使用时间嵌入向量")
parser.add_argument("--spatial_emb", type=eval, default=config['model']['spatial_emb'], help="是否使用空间嵌入向量")
parser.add_argument("--use_mask", type=eval, default=config['model']['use_mask'], help="是否使用mask矩阵优化adj")
parser.add_argument("--activation", type=str, default=config['model']['activation'], help="激活函数 {relu, GlU}")

parser.add_argument('--seed', type=int, default=config['train']['seed'], help='种子设置')
parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'], help="初始学习率")
parser.add_argument("--lr_decay", type=eval, default=config['train']['lr_decay'], help="是否开启初始学习率衰减策略")
parser.add_argument("--lr_decay_step", type=str, default=config['train']['lr_decay_step'], help="在几个epoch进行初始学习率衰减")
parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'], help="学习率衰减率")
parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help="训练代数")
parser.add_argument('--print_every', type=int, default=config['train']['print_every'], help='几个batch报训练损失')
parser.add_argument('--save', type=str, default=config['train']['save'], help='保存路径')
parser.add_argument('--expid', type=int, default=config['train']['expid'], help='实验 id')
parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'], help="梯度阈值")

parser.add_argument('--patience', type=int, default=config['train']['patience'], help='等待代数')
parser.add_argument('--log_file', default=config['train']['log_file'], help='log file')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("cuda:",device)


log = open(args.log_file, 'w')
log_string(log, str(args))


def main():
    # load data
    adj = get_adjacency_matrix(distance_df_filename=args.sensors_distance,
                               num_of_vertices=args.num_of_vertices,
                               type_=args.construct_type,
                               id_filename=None)
    local_adj = construct_adj(A=adj, steps=args.strides)
    local_adj = torch.FloatTensor(local_adj)

    dataloader = load_dataset(dataset_dir=args.data,
                              normalizer=args.normalizer,
                              batch_size=args.batch_size,
                              valid_batch_size=args.batch_size,
                              test_batch_size=args.batch_size,
                              column_wise=args.column_wise)

    scaler = dataloader['scaler']

    log_string(log, 'loading data...')

    log_string(log, "The shape of localized adjacency matrix: {}".format(local_adj.shape))

    log_string(log, f'trainX: {torch.tensor(dataloader["train_loader"].xs).shape}\t\t '
                    f'trainY: {torch.tensor(dataloader["train_loader"].ys).shape}')
    log_string(log, f'valX:   {torch.tensor(dataloader["val_loader"].xs).shape}\t\t'
                    f'valY:   {torch.tensor(dataloader["val_loader"].ys).shape}')
    log_string(log, f'testX:   {torch.tensor(dataloader["test_loader"].xs).shape}\t\t'
                    f'testY:   {torch.tensor(dataloader["test_loader"].ys).shape}')
    log_string(log, f'mean:   {scaler.mean:.4f}\t\tstd:   {scaler.std:.4f}')
    log_string(log, 'data loaded!')

    engine = trainer(args=args,
                     scaler=scaler,
                     adj=local_adj,
                     history=args.history,
                     num_of_vertices=args.num_of_vertices,
                     in_dim=args.in_dim,
                     hidden_dims=args.hidden_dims,
                     first_layer_embedding_size=args.first_layer_embedding_size,
                     out_layer_dim=args.out_layer_dim,
                     log=log,
                     lrate=args.learning_rate,
                     device=device,
                     activation=args.activation,
                     use_mask=args.use_mask,
                     max_grad_norm=args.max_grad_norm,
                     lr_decay=args.lr_decay,
                     temporal_emb=args.temporal_emb,
                     spatial_emb=args.spatial_emb,
                     horizon=args.horizon,
                     strides=args.strides)

    # 开始训练
    log_string(log, 'compiling model...')
    his_loss = []
    val_time = []
    train_time = []
    #
    wait = 0
    val_loss_min = float('inf')
    # val_loss_min = 19.34
    best_model_wts = None

    for i in tqdm.tqdm(range(1, args.epochs + 1)):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {i:04d}')
            break

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            # [B, T, N, C]

            trainy = torch.Tensor(y[:, :, :, 0]).to(device)
            # [B, T, N]

            loss, tmae, tmape, trmse = engine.train(trainx, trainy)
            train_loss.append(loss)
            train_mae.append(tmae)
            train_mape.append(tmape)
            train_rmse.append(trmse)

            if iter % args.print_every == 0:
                logs = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, lr: {}'
                print(logs.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1],
                                  engine.optimizer.param_groups[0]['lr']), flush=True)

        if args.lr_decay:
            engine.lr_scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).to(device)
            # [B, T, N, C]

            valy = torch.Tensor(y[:, :, :, 0]).to(device)
            # [B, T, N]

            vmae, vmape, vrmse = engine.evel(valx, valy)
            valid_loss.append(vmae)
            valid_mape.append(vmape)
            valid_rmse.append(vrmse)

        s2 = time.time()
        logs = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        log_string(log, logs.format(i, (s2-s1)))

        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        logs = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        log_string(log, logs.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))
        # os.system('nvidia-smi')

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if mvalid_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {mvalid_loss:.4f}, '
                f'save model to {args.save + "exp_" + str(args.expid) + "_" + str(round(mvalid_loss, 2)) + "_best_model.pth"}'
            )
            wait = 0
            val_loss_min = mvalid_loss
            best_model_wts = engine.model.state_dict()
            torch.save(best_model_wts,
                       args.save + "exp_" + str(args.expid) + "_" + str(round(val_loss_min, 2)) + "_best_model.pth")
        else:
            wait += 1

        np.save('./history_loss' + f'_{args.expid}', his_loss)

    log_string(log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string(log, "Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # 测试
    engine.model.load_state_dict(
        torch.load(args.save + "exp_" + str(args.expid) + "_" + str(round(val_loss_min, 2)) + "_best_model.pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test'][:, :, :, 0]).to(device)
    # B, T, N

    for iter, (x, y) in tqdm.tqdm(enumerate(dataloader['test_loader'].get_iterator())):
        testx = torch.Tensor(x).to(device)
        with torch.no_grad():
            preds = engine.model(testx)
            # [B, T, N]
            outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  # 这里是因为你在做batch的时候，可能会padding出新的sample以满足batch_size的要求
    # 保存预测结果为 numpy 文件
    np.save(f"{args.save}exp_{args.expid}_predictions.npy", yhat.cpu().numpy())
    # np.save(f"{args.save}exp_{args.expid}_predictions.npy", yhat.cpu().numpy())
    np.savez(f"{args.save}exp_{args.expid}_results.npz",
             ground_truth=realy.cpu().numpy(),
             predictions=yhat.cpu().numpy())

    # 预测结果和真实值的保存，ground_truth为真实值，predictions为预测值，两者的形状相同。

    log_string(log, "Training finished")
    log_string(log, "The valid loss on best model is " + str(round(val_loss_min, 4)))

    amae = []
    amape = []
    armse = []

    for t in range(args.horizon):
        pred = scaler.inverse_transform(yhat[:, t, :])
        real = realy[:, t, :]

        mae, mape, rmse = metric(pred, real)
        logs = '最好的验证模型在测试集上对 horizon: {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'

        log_string(log, logs.format(t+1, mae, mape, rmse))
        amae.append(mae)
        amape.append(mape*100)
        armse.append(rmse)

    logs = '总平均测试结果, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    log_string(log, logs.format(np.mean(amae), np.mean(amape), np.mean(armse)))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()

# D:\Anoconde\envs\STGCN\python.exe D:\基线方法\STSGCN\train.py
# cuda: cuda:0
# Namespace(no_cuda=True, data='./data/processed/PEMS08/', sensors_distance='./data/PEMS08/PEMS08.csv', column_wise=False, normalizer='std', batch_size=32, num_of_vertices=170, construct_type='connectivity', in_dim=1, hidden_dims=[[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]], first_layer_embedding_size=64, out_layer_dim=128, history=12, horizon=12, strides=3, temporal_emb=True, spatial_emb=True, use_mask=True, activation='GLU', seed=10, learning_rate=0.003, lr_decay=True, lr_decay_step='15,40,70,105,145', lr_decay_rate=0.3, epochs=1, print_every=50, save='./garage/PEMSD8/', expid=1, max_grad_norm=5.0, patience=20, log_file='./data/log_PEMSD8')
# Normalize the dataset by Standard Normalization
# loading data...
# The shape of localized adjacency matrix: torch.Size([510, 510])
# trainX: torch.Size([10720, 12, 170, 1])		 trainY: torch.Size([10720, 12, 170, 1])
# valX:   torch.Size([3584, 12, 170, 1])		valY:   torch.Size([3584, 12, 170, 1])
# testX:   torch.Size([3584, 12, 170, 1])		testY:   torch.Size([3584, 12, 170, 1])
# mean:   229.8431		std:   145.6255
# data loaded!
# Applying learning rate decay.
# 模型可训练参数: 1,401,232
# GPU使用情况:5.612032
# compiling model...
#   0%|          | 0/1 [00:00<?, ?it/s]Iter: 000, Train Loss: 146.5736, Train MAE: 147.0729, Train MAPE: 3.3127, Train RMSE: 175.0359, lr: 0.003
# Iter: 050, Train Loss: 87.7477, Train MAE: 88.2465, Train MAPE: 0.9517, Train RMSE: 115.3596, lr: 0.003
# Iter: 100, Train Loss: 82.5229, Train MAE: 83.0214, Train MAPE: 0.9521, Train RMSE: 104.1624, lr: 0.003
# Iter: 150, Train Loss: 40.9120, Train MAE: 41.4088, Train MAPE: 0.6185, Train RMSE: 57.4528, lr: 0.003
# Iter: 200, Train Loss: 35.0743, Train MAE: 35.5702, Train MAPE: 0.3967, Train RMSE: 53.1774, lr: 0.003
# Iter: 250, Train Loss: 32.1459, Train MAE: 32.6418, Train MAPE: 0.2396, Train RMSE: 47.4318, lr: 0.003
# Iter: 300, Train Loss: 27.4307, Train MAE: 27.9258, Train MAPE: 0.2088, Train RMSE: 40.6218, lr: 0.003
# Epoch: 001, Inference Time: 9.6686 secs
# Epoch: 001, Train Loss: 55.0084, Train MAE 55.5053, Train MAPE: 0.7048, Train RMSE: 74.0516, Valid Loss: 30.3137, Valid MAPE: 0.2806, Valid RMSE: 43.2105, Training Time: 89.8336/epoch
# val loss decrease from inf to 30.3137, save model to ./garage/PEMSD8/exp_1_30.31_best_model.pth
# Average Training Time: 89.8336 secs/epoch
# Average Inference Time: 9.6686 secs
# 100%|██████████| 1/1 [01:39<00:00, 99.54s/it]
# 112it [00:09, 12.34it/s]
# Training finished
# The valid loss on best model is 30.3137
# 最好的验证模型在测试集上对 horizon: 1, Test MAE: 26.1283, Test MAPE: 0.2087, Test RMSE: 37.4734
# 最好的验证模型在测试集上对 horizon: 2, Test MAE: 26.1557, Test MAPE: 0.2101, Test RMSE: 37.6972
# 最好的验证模型在测试集上对 horizon: 3, Test MAE: 27.3315, Test MAPE: 0.2143, Test RMSE: 38.8559
# 最好的验证模型在测试集上对 horizon: 4, Test MAE: 27.0166, Test MAPE: 0.2113, Test RMSE: 38.7420
# 最好的验证模型在测试集上对 horizon: 5, Test MAE: 27.1706, Test MAPE: 0.2046, Test RMSE: 39.4165
# 最好的验证模型在测试集上对 horizon: 6, Test MAE: 28.2399, Test MAPE: 0.2274, Test RMSE: 40.3142
# 最好的验证模型在测试集上对 horizon: 7, Test MAE: 28.9740, Test MAPE: 0.2338, Test RMSE: 41.2641
# 最好的验证模型在测试集上对 horizon: 8, Test MAE: 30.0176, Test MAPE: 0.2486, Test RMSE: 42.5381
# 最好的验证模型在测试集上对 horizon: 9, Test MAE: 31.1500, Test MAPE: 0.2540, Test RMSE: 44.1428
# 最好的验证模型在测试集上对 horizon: 10, Test MAE: 33.1150, Test MAPE: 0.2580, Test RMSE: 47.4235
# 最好的验证模型在测试集上对 horizon: 11, Test MAE: 33.2950, Test MAPE: 0.2644, Test RMSE: 47.1694
# 最好的验证模型在测试集上对 horizon: 12, Test MAE: 36.5099, Test MAPE: 0.3885, Test RMSE: 49.8425
# 总平均测试结果, Test MAE: 29.5920, Test MAPE: 24.3632, Test RMSE: 42.0733
# total time: 1.9min
#
# 进程已结束，退出代码为 0
