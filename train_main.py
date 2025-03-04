# -*- encoding: utf-8 -*-

import os
import argparse
import datetime
from loguru import logger
import numpy as np
import pickle
import torch
from sklearn.cluster import KMeans

from model import AE_NN, FULL_NN, ClusterAssignment
from utils import evaluation, get_laplace_matrix
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

def sinkhorn(pred, lambdas, row, col):
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    p = np.power(pred, lambdas)
    
    u = np.ones(num_node)
    v = np.ones(num_class)

    for index in range(1000):
        u = row * np.power(np.dot(p, v), -1)
        u[np.isinf(u)] = -9e-15
        v = col * np.power(np.dot(u, p), -1)
        v[np.isinf(v)] = -9e-15
    u = row * np.power(np.dot(p, v), -1)
    target = np.dot(np.dot(np.diag(u), p), np.diag(v))
    return target

if __name__ == "__main__":
    torch.set_num_threads(8)

    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--dataname', default='acm', type=str)
    parser.add_argument('--gpu', default=0, type=int)

    embedding_num = 16
    parser.add_argument('--dims_encoder', default=[256, embedding_num], type=list)
    parser.add_argument('--dims_decoder', default=[embedding_num, 256], type=list)

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lambdas', default=5, type=float)
    parser.add_argument('--balancer', default=0.5, type=float)

    parser.add_argument('--factor_ort', default=1, type=float)
    parser.add_argument('--factor_KL', default=0.5, type=float)
    parser.add_argument('--factor_corvar', default=0.05, type=float)

    parser.add_argument('--pretrain_model_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_centers_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_pseudo_labels_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_model_load_path', default='pkl', type=str)
    parser.add_argument('--pretrain_centers_load_path', default='pkl', type=str)
    parser.add_argument('--pretrain_pseudo_labels_load_path', default='pkl', type=str)

    parser.add_argument('--foldername', default='MAIN_modified', type=str)
    parser.add_argument('--noramlize_flag', default=False, type=bool)

    args = parser.parse_args()
    args.pretrain_model_save_path = './result/{}/{}_model.pkl'.format(args.foldername, args.dataname)
    args.pretrain_centers_save_path = './result/{}/{}_centers.pkl'.format(args.foldername, args.dataname)
    args.pretrain_pseudo_labels_save_path = './result/{}/{}_pseudo_labels.pkl'.format(args.foldername, args.dataname)
    args.pretrain_model_load_path = './result/{}/{}_model.pkl'.format(args.foldername, args.dataname)
    args.pretrain_centers_load_path = './result/{}/{}_centers.pkl'.format(args.foldername, args.dataname)
    args.pretrain_pseudo_labels_load_path = './result/{}/{}_pseudo_labels.pkl'.format(args.foldername, args.dataname)

    if os.path.isdir('./result/{}/'.format(args.foldername)) == False:
        os.makedirs('./result/{}/'.format(args.foldername))
    if os.path.isdir('./log/{}/'.format(args.foldername)) == False:
        os.makedirs('./log/{}/'.format(args.foldername))

    if args.dataname == 'acm':
        args.num_class = 3
        args.learning_rate = 5e-3
        args.weight_decay = 5e-3
        args.alpha_pre = 0.3
        args.eta = 1e-5
        args.alpha = 0.7
        args.beta = 0.5
        args.balancer = 0.5
        args.factor_ort = 1
        args.factor_KL = 0.1
        args.factor_corvar = 0.25
        args.factor_construct = 0.0
        args.graph_path = './data/{}_graph.txt'.format(args.dataname)
    if args.dataname == 'cite':
        args.num_class = 6
        args.learning_rate = 5e-3
        args.weight_decay = 5e-3
        args.alpha_pre = 0.01
        args.eta = 1e-4
        args.alpha = 0.3
        args.beta = 0.1
        args.balancer = 0.4
        args.factor_ort = 0.6 
        args.factor_KL = 0.1
        args.factor_corvar = 0.4
        args.factor_construct = 0.0
        args.graph_path = './data/{}_graph.npy'.format(args.dataname)
    if args.dataname == 'dblp':
        args.num_class = 4
        args.learning_rate = 1e-3
        args.weight_decay = 5e-3
        args.alpha_pre = 0.1
        args.eta = 1e-4
        args.alpha = 0.1
        args.beta = 3
        args.balancer = 0.8  
        args.factor_KL = 0.1
        args.factor_ort = 0.5
        args.factor_corvar = 0.1
        args.factor_construct = 0.0
        args.graph_path = './data/{}_graph.npy'.format(args.dataname)
    if args.dataname == 'amazon':
        args.num_class = 8
        args.learning_rate = 1e-3
        args.weight_decay = 5e-3
        args.alpha_pre = 0.1
        args.eta = 1e-4
        args.alpha = 0.1
        args.beta = 3
        args.balancer = 0.9
        args.factor_ort = 0.8
        args.factor_KL = 0.1
        args.factor_corvar = 0.2
        args.lambdas = 20
        args.noramlize_flag = True
        args.factor_construct = 0.0
        args.graph_path = './data/{}_graph.txt'.format(args.dataname)
    if args.dataname == 'hhar':
        args.num_class = 6
        args.learning_rate = 1e-3
        args.weight_decay = 1e-3
        args.alpha_pre = 3.0
        args.eta = 1e-5
        args.alpha = 0.7
        args.beta = 0.5
        args.factor_ort = 1
        args.factor_KL = 0.1
        args.factor_corvar = 0.05
        args.factor_construct = 0.0
        args.graph_path = './data/{}5_graph.txt'.format(args.dataname)
    if args.dataname == 'reut':
        args.num_class = 4
        args.learning_rate = 5e-3
        args.weight_decay = 1e-3
        args.alpha_pre = 0.01
        args.eta = 1e-4
        args.alpha = 0.001
        args.beta = 0.1
        args.factor_ort = 1
        args.factor_KL = 0.1
        args.factor_corvar = 0.05
        args.factor_construct = 0.0
        args.graph_path = './data/{}5_graph.txt'.format(args.dataname)   
    if args.dataname == 'usps':
        args.num_class = 10
        args.learning_rate = 1e-4
        args.weight_decay = 5e-4
        args.alpha_pre = 1000
        args.eta = 1e-3
        args.alpha = 0.5
        args.beta = 5
        args.factor_ort = 1
        args.factor_KL = 0.1
        args.factor_corvar = 0.05
        args.factor_construct = 0.0
        args.graph_path = './data/{}5_graph.txt'.format(args.dataname)  
    if args.dataname == 'cora':
        args.num_class = 7
        args.learning_rate = 1e-3
        args.weight_decay = 5e-4
        args.alpha_pre = 1000
        args.balancer = 0.9
        args.eta = 1e-3
        args.alpha = 0.5
        args.beta = 5
        args.factor_ort = 20
        args.factor_KL = 2
        args.factor_corvar = 1
        args.factor_construct = 0.0
        args.graph_path = './data/{}_graph.npy'.format(args.dataname)
    if args.dataname == 'corafull':
        args.num_class = 70
        args.learning_rate = 1e-3
        args.weight_decay = 5e-4
        args.alpha_pre = 1000
        args.balancer = 0.9
        args.eta = 1e-3
        args.alpha = 0.5
        args.beta = 5
        args.factor_ort = 20
        args.factor_KL = 2
        args.factor_corvar = 1
        args.factor_construct = 0.0
        args.graph_path = './data/{}_graph.npy'.format(args.dataname)
    if args.dataname == 'pubmed':
        args.num_class = 3
        args.learning_rate = 1e-2
        args.weight_decay = 5e-3
        args.alpha_pre = 1000
        args.balancer = 0.9
        args.eta = 1e-3
        args.alpha = 0.5
        args.beta = 5
        args.factor_ort = 2
        args.factor_KL = 0.5
        args.factor_corvar = 1
        args.factor_construct = 0.0
        args.lambdas = 20
        args.graph_path = './data/{}_graph.npy'.format(args.dataname)

    now = datetime.datetime.now()
    timestamp = now.strftime("%y-%m-%dT%H:%M:%S")
    logger.add('./log/{}/{}/{}.log'.format(args.foldername, args.dataname, timestamp), rotation="500 MB", level="INFO")
    logger.info(args)
    logger.info('dataname: {}'.format(args.dataname))
    logger.info('lr: {} | balancer: {} | factor_ort: {} | factor_corvar: {} | factor_KL: {} | factor_construct: {} | lambads: {} | weight_decay: {}'.format(args.learning_rate, args.balancer, args.factor_ort, args.factor_corvar, args.factor_KL, args.factor_construct, args.lambdas, args.weight_decay))
    logger.info('dims_encoder: {} | dims_decoder: {}'.format(args.dims_encoder, args.dims_decoder))
    logger.info('###############################################')

    torch.cuda.set_device(args.gpu)
    # load x
    if args.dataname in ['acm', 'amazon']:
        x = torch.tensor(np.loadtxt('./data/{}.txt'.format(args.dataname), dtype=float), dtype=torch.float)
    else:
        x = torch.tensor(np.load('./data/{}_feat.npy'.format(args.dataname)), dtype=torch.float)
    ## normalize
    if args.dataname in ['amazon'] and args.noramlize_flag == True:
        x = torch.tensor(x, dtype=torch.float)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        print("normalize done!")
    x_ = torch.nn.functional.normalize(x, p=2, dim=1)
    # load y
    if args.dataname in ['acm', 'amazon']:
        y = torch.tensor(np.loadtxt('./data/{}_label.txt'.format(args.dataname), dtype=int), dtype=torch.float)
    elif args.dataname in ['cite', 'hhar']:
        y = torch.tensor(np.load('./data/{}_label.npy'.format(args.dataname)), dtype=torch.float)-1
    else:
        y = torch.tensor(np.load('./data/{}_label.npy'.format(args.dataname)), dtype=torch.float)
    # load edge
    if args.dataname in ['acm', 'amazon']:
        edge_index = torch.tensor(np.loadtxt(args.graph_path, dtype=int), dtype=torch.long).T
    else:
        edge_index = torch.tensor(np.load(args.graph_path), dtype=torch.long).T
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), [x.shape[0], x.shape[0]]).to_dense()
    # symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # add self-loop
    adj_self_loop = adj+torch.eye(x.shape[0])

    if args.dataname == 'hhar':
        def heat_kernel(x_1, x_2):
            return torch.exp(-0.01 * torch.pow(torch.cdist(x_1, x_2, p=2), 2))
        adj_f = heat_kernel(x, x)
    else:
        x_ = torch.nn.functional.normalize(x, p=2, dim=1)
        adj_f = torch.mm(x_, x_.T)
    
    L_1 = get_laplace_matrix(adj_self_loop)
    L_2 = get_laplace_matrix(adj_f)

    best_metrics = {'acc': [], 'nmi': [], 'ari': [], 'f1': []}
    for seed in [100]:
        logger.info('Seed {}'.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # ################################### PRE-TRAIN ########################################
        Model = AE_NN(dim_input=x.shape[1], dims_encoder=args.dims_encoder, dims_decoder=args.dims_decoder).cuda()
        optimizer = torch.optim.Adam(Model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        acc_max = 0

        for epoch in range(1, args.epochs+1):

            h, x_hat = Model.forward(x.cuda(), adj_self_loop.cuda())
            z = torch.nn.functional.normalize(h, p=2, dim=0)
            adj_pred = torch.mm(z, z.T)

            loss_x = torch.nn.functional.mse_loss(x_hat, x.cuda())
            loss_corvariates = -torch.mm(torch.mm(z.T, (args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),z).trace()/len(z.T)
            loss_ort =  torch.nn.functional.mse_loss(torch.mm(z.T,z).view(-1).cuda(),torch.eye(len(z.T)).view(-1).cuda())
            loss = args.factor_construct * loss_x + args.factor_ort * loss_ort + args.factor_corvar * loss_corvariates

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                kmeans = KMeans(n_clusters=args.num_class, random_state=2021, n_init=20).fit(z.cpu().numpy())
                acc, nmi, ari, f1_macro = evaluation(y, kmeans.labels_)
                centers = torch.tensor(kmeans.cluster_centers_)

                if epoch % 10 == 0:
                    logger.info('Epoch {}/{} | loss_corvariate: {:.6f} | loss_ort: {:.6f} | loss_total: {:.6f}'.format(epoch, args.epochs, loss_corvariates.cpu().item(), loss_ort.cpu().item(),  loss.cpu().item()))
                    logger.info('Epoch {}/{} | Pre-Train ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(epoch, args.epochs, acc, nmi, ari, f1_macro))

                if acc > acc_max:
                    acc_max = acc
                    torch.save(Model.state_dict(), args.pretrain_model_save_path)
                    with open(args.pretrain_centers_save_path,'wb') as save1:
                        pickle.dump(centers, save1, protocol=pickle.HIGHEST_PROTOCOL)
                    pseudo_labels = torch.LongTensor(kmeans.labels_)
                    with open(args.pretrain_pseudo_labels_save_path,'wb') as save2:
                        pickle.dump(pseudo_labels, save2, protocol=pickle.HIGHEST_PROTOCOL)
    
        ####################################### TRAIN ########################################
        Model = FULL_NN(dim_input=x.shape[1], dims_encoder=args.dims_encoder, dims_decoder=args.dims_decoder, num_class=args.num_class, \
                    pretrain_model_load_path=args.pretrain_model_load_path).cuda()
        optimizer = torch.optim.Adam(Model.parameters(), lr=args.learning_rate)

        with open(args.pretrain_centers_load_path,'rb') as load1:
            centers = pickle.load(load1).cuda()
        with open(args.pretrain_pseudo_labels_load_path,'rb') as load2:
            pseudo_labels = pickle.load(load2).cuda()

        acc_max, nmi_max, ari_max, f1_macro_max = 0, 0, 0, 0
        best_epoch = 0
        for epoch in range(1, args.epochs+1):
            z, x_hat = Model.forward(x.cuda(), adj_self_loop.cuda())
            z = torch.nn.functional.normalize(z, p=2, dim=0)
            centers = centers.detach()

            adj_pred = torch.mm(z, z.T)
            loss_x = torch.nn.functional.mse_loss(x_hat, x.cuda())

            loss_corvariates = -torch.mm(torch.mm(z.T, ( args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),z).trace()/len(z.T)
            loss_ort = torch.nn.functional.mse_loss(torch.mm(z.T,z).view(-1).cuda(),torch.eye(len(z.T)).view(-1).cuda())
            loss_adj_graph = torch.nn.functional.mse_loss(adj_pred.view(-1), adj_self_loop.cuda().view(-1))
       
            class_assign_model = ClusterAssignment(args.num_class, len(z.T), 1, centers).cuda()
            temp_class = class_assign_model(z.cuda())
            
            ### sinkhole
            if epoch == 1:
                p_distribution = torch.tensor(sinkhorn ( temp_class.cpu().detach().numpy(), args.lambdas, torch.ones(x.shape[0]).numpy(), torch.tensor([torch.sum(pseudo_labels==i) for i in range(args.num_class)]).numpy())).float().cuda().detach()
                p_distribution = p_distribution.detach()
                q_max, q_max_index = torch.max(p_distribution, dim=1)
            elif epoch // 10 == 0:
                p_distribution = torch.tensor(sinkhorn ( temp_class.cpu().detach().numpy(), args.lambdas, torch.ones(x.shape[0]).numpy(), torch.tensor([torch.sum(pseudo_labels==i) for i in range(args.num_class)]).numpy())).float().cuda().detach()
                p_distribution = p_distribution.detach()
                q_max, q_max_index = torch.max(p_distribution, dim=1)

            KL_loss_function = nn.KLDivLoss(reduction='sum')
            loss_KL = KL_loss_function(temp_class.cuda(), p_distribution.cuda()) / temp_class.shape[0]

            loss = args.factor_construct * loss_x + args.factor_ort * loss_ort + args.factor_corvar * loss_corvariates + args.factor_KL * loss_KL

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                kmeans = KMeans(n_clusters=args.num_class, random_state=2021, n_init=20).fit(z.cpu().numpy())
                acc, nmi, ari, f1_macro = evaluation(y, kmeans.labels_)
                if acc_max < acc:
                    acc_max, nmi_max, ari_max, f1_macro_max = acc, nmi, ari, f1_macro
                    best_epoch = epoch
                pseudo_labels = torch.LongTensor(kmeans.labels_)
                centers = torch.tensor(kmeans.cluster_centers_)
                #### logger
                logger.info('Epoch {}/{} | loss_corvariate: {:.6f} | loss_ort: {:.6f} | loss_KL: {:.6f} | loss_total: {:.6f}'.format(epoch, args.epochs, loss_corvariates.cpu().item(), loss_ort.cpu().item(), loss_KL.cpu().item(),  loss.cpu().item()))
                logger.info('Epoch {}/{} | ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(epoch, args.epochs, acc, nmi, ari, f1_macro))
        logger.info('MAX at {} | ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f} | Seed: {}'.format(best_epoch, acc_max, nmi_max, ari_max, f1_macro_max, seed))
        best_metrics['acc'].append(acc_max)
        best_metrics['nmi'].append(nmi_max)
        best_metrics['ari'].append(ari_max)
        best_metrics['f1'].append(f1_macro_max)
    logger.info('###############################################')
    logger.info('            | ACC\t\tNMI\t\tARI\t\tF1')
    logger.info('Mean of Max | {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(np.mean(best_metrics['acc']), np.mean(best_metrics['nmi']), np.mean(best_metrics['ari']), np.mean(best_metrics['f1'])))
    logger.info('Std. of MAX | {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(np.std(best_metrics['acc']), np.std(best_metrics['nmi']), np.std(best_metrics['ari']), np.std(best_metrics['f1'])))
        