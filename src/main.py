import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from dataset.wm811k import WM811K
from utils.dataloader import *
from utils.contrastive_loss import SupConLoss
from utils.utils import *

from classifier.models.rnn import RNN
from classifier.models.resnet import ResNet, SupConResNet
from confidnet.models.rnn_selfconfid import RNNSelfConfid


from classifier.trainer.train_cnn import training_loop
from classifier.trainer.train_rnn import training_loop_rnn
from confidnet.trainer.train_confidnet import training_loop_confidnet
from patch_selector.trainer.train_reinforce import training_loop_reinforce

from confidnet.utils.confidnet_loss import RNNSelfConfidMSELoss
from confidnet.utils.utils import accuracy_rnnconfidnet

from patch_selector.models.my_rnn import Classifier
from patch_selector.models.confidnet import Confidnet

from patch_selector.models.reinforce import REINFORCEAgent
from patch_selector.models.environment import WaferMapGrid
from patch_selector.models.reinforce_ensemble import REINFORCEAgentEnsemble
from patch_selector.models.environment_ensemble import WaferMapGridEnsemble
from patch_selector.utils.eval import eval_reinforce

import parser



TYPE_TO_CLASS = {
    'Loc' : 0, 
    'Edge-Loc' : 1, 
    'Center' : 2, 
    'Edge-Ring' : 3, 
    'Scratch' : 4,
    'Random' : 5, 
    'Near-full' : 6, 
    'Donut' : 7,
    'none' : 8,
}



def main():
    # ---------------------------------------------------------------------------- Hyperparameters reading ----------------------------------------------------------------------------
    args = parser.get_parser().parse_args()
    args = vars(args)
    
    SEED = args["seed"]
    data_name = args["data_name"]
    data_path = args["data_path"]
    input_channels = args["input_channels"]
    patch_size = args["patch_size"]
    
    TRAIN_CLASSIFIER = args["train_classifier"]
    classifier_name = args["classifier_name"]
    classifier_hidden_dim = args["classifier_hidden_dim"]
    classifier_num_layers = args["classifier_num_layers"]
    supcon = args["supcon"]
    
    classifier_optimizer_name = args["classifier_optimizer_name"]
    classifier_num_epochs = args["classifier_num_epochs"]
    classifier_lr = args["classifier_lr"]
    classifier_wd = args["classifier_wd"]
    classifier_batch_size = args["classifier_batch_size"]
    classifier_num_heads = args["classifier_num_heads"]
    
    img_size = (64, 64)
    permute_patches = True
    num_classes = 8 # we exclude 'none' class from consideration
    seq_len = int(img_size[0] * img_size[1] // patch_size ** 2)
    classifier_input_dim = input_channels * patch_size ** 2
    
    seed_everything(seed=SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    
    HYBRID = args["hybrid"]    
    ENSEMBLE = args["ensemble"]
    
    # ---------------------------------------------------------------------------- DATA PREPARATION ----------------------------------------------------------------------------
    
    train_dataset = load_dataset(data_name=data_name, path=data_path, train=True, supcon=supcon,
                                 input_channels=input_channels, img_size=img_size, num_classes=num_classes)
    
    test_dataset = load_dataset(data_name=data_name, path=data_path, train=False, supcon=False, 
                                input_channels=input_channels, img_size=img_size, num_classes=num_classes)
    
    val_idx = torch.arange(0, len(test_dataset), 2)
    test_idx = torch.tensor([i for i in range(len(test_dataset)) if i not in val_idx])
    
    val_dataset = load_dataset(data_name=data_name, path=data_path, train=False, val=True, 
                               input_channels=input_channels, img_size=img_size, num_classes=num_classes, idx=val_idx)
    
    test_dataset.df = test_dataset.df.iloc[test_idx].reset_index()
    train_loader, test_loader, val_loader = get_loaders(train_dataset, test_dataset, val_dataset=val_dataset, batch_size=classifier_batch_size)    
    # ---------------------------------------------------------------------------- CLASSIFIER TRAINING & EVALUATION ----------------------------------------------------------------------------
    if not ENSEMBLE:
        if "resnet" in classifier_name:
            if supcon:
                classifier = SupConResNet(model_name=classifier_name, num_classes=num_classes, 
                                          input_channels=input_channels,  feat_dim=128, pretrained=False)
            else:
                classifier = ResNet(model_name=classifier_name, num_classes=num_classes, 
                                    input_channels=input_channels, pretrained=False)
        else:
            classifier = RNN(input_dim=classifier_input_dim, hidden_dim=classifier_hidden_dim, num_classes=num_classes, seq_len=seq_len, 
                             num_layers=classifier_num_layers, patch_size=patch_size, cell_type=classifier_name, permute_patches=permute_patches)

        classifier = classifier.to(device)

        PATH_classifier = f"{classifier_name}_seed{SEED}.pth"
        if TRAIN_CLASSIFIER:
            criterion = choose_criterion()
            optimizer = choose_optimizer(classifier, classifier_optimizer_name, lr=classifier_lr, wd=classifier_wd)

            if supcon:
                supcon_loss = SupConLoss(
                                num_classes=8,
                    )
            else:
                supcon_loss = None

            if classifier_name == "gru":
                scheduler = None
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.33333, verbose=False)

            if "resnet" in classifier_name:
                classifier, _ = training_loop(classifier, criterion, optimizer, scheduler, 
                                              train_loader, val_loader, classifier_num_epochs, device, supcon_loss=supcon_loss,
                                              file_name=PATH_classifier)
            elif 'gru' in classifier_name or 'lstm' in classifier_name:
                classifier, _ = training_loop_rnn(classifier, criterion, optimizer, scheduler, 
                                                  train_loader, val_loader, classifier_num_epochs, classifier_num_heads, device,
                                                  file_name=PATH_classifier)

        classifier.load_state_dict(torch.load(PATH_classifier))
        seed_everything(seed=SEED)
        if 'resnet' in classifier_name:
            test_acc = accuracy_cnn(classifier, test_loader, device=device, supcon=supcon)
            f1 = f1score_cnn(classifier, test_loader, device, supcon=supcon)
        else:
            test_acc = accuracy(classifier, test_loader, device=device)
            f1 = f1score(classifier, test_loader, device)

        print("Classifier accuracy: ", 100*test_acc)
        print("Classifier F1 score: ", f1)
        print("------------------")
    
    # ---------------------------------------------------------------------------- CONFIDNET TRAINING & EVALUATION ----------------------------------------------------------------------------
    
    TRAIN_CONFIDNET = args["train_confidnet"]
    confidnet_name = args["confidnet_name"]
    confidnet_hidden_dim = args["confidnet_hidden_dim"]
    confidnet_num_layers = args["confidnet_num_layers"]

    confidnet_optimizer_name = args["confidnet_optimizer_name"]
    confidnet_num_epochs = args["confidnet_num_epochs"]
    confidnet_lr = args["confidnet_lr"]
    confidnet_wd = args["confidnet_wd"]
    confidnet_batch_size = args["confidnet_batch_size"]
            
    if "resnet" not in classifier_name and not ENSEMBLE:
        confidnet = RNNSelfConfid(input_dim=classifier_hidden_dim, hidden_dim=confidnet_hidden_dim, seq_len=seq_len, num_layers=confidnet_num_layers, batch_first=True).to(device)

        PATH_confidnet = f"confid{confidnet_name}_seed{SEED}.pth"
        if TRAIN_CONFIDNET:
            criterion = RNNSelfConfidMSELoss(num_classes=num_classes, weight=10)
            optimizer = choose_optimizer(confidnet, confidnet_optimizer_name, lr=confidnet_lr, wd=confidnet_wd)
            scheduler = None

            confidnet, _ = training_loop_confidnet(confidnet, classifier, criterion, optimizer, scheduler, 
                                                   val_loader, test_loader, confidnet_num_epochs, device,
                                                   file_name=PATH_confidnet)
        confidnet.load_state_dict(torch.load(PATH_confidnet))
            
        if HYBRID:
            aux_model_name = args["aux_model_name"]
            PATH_aux_model = f"{aux_model_name}_seed{SEED}.pth"
            if supcon:
                aux_model = SupConResNet(model_name=aux_model_name, num_classes=num_classes, 
                                         input_channels=input_channels,  feat_dim=128, pretrained=False)
            else:
                aux_model = ResNet(model_name=aux_model_name, num_classes=num_classes, 
                                   input_channels=input_channels, pretrained=False)
            
            aux_model = aux_model.to(device)
            aux_model.load_state_dict(torch.load(PATH_aux_model))
        else:
            aux_model = None

        acc_conf = 0
        n_scans = []
        f1_conf = 0
        num_rep = 10
        seed_everything(seed=SEED)
        for _ in range(num_rep):
            acc, f1, idx, num_scans = accuracy_rnnconfidnet(classifier, confidnet, test_loader, device, cnn_model=aux_model, patch_order=None, 
                                                            prediction_index='confidnet', offset1=0, offset2=seq_len, supcon=supcon)
            acc_conf += 100 * acc
            f1_conf += f1
            n_scans.append(num_scans)

        n_scans = torch.cat(n_scans)
        acc_conf /= num_rep
        f1_conf /= num_rep

        print("Classifier + Confidnet accuracy: ", acc_conf)
        print("Classifier + Confidnet F1 score: ", f1_conf)
        print("------------------")
        
    # ---------------------------------------------------------------------------- PATCH SELCTOR TRAINING & EVALUATION ----------------------------------------------------------------------------
    
    TRAIN_PATCH_SELECTOR = args["train_patch_selector"]
    patch_selector_num_iters = args["patch_selector_num_iters"]
    patch_selector_lr = args["patch_selector_lr"]
    patch_selector_wd = args["patch_selector_wd"]
    patch_selector_batch_size = args["patch_selector_batch_size"]
    gamma = args["gamma"]
    chkpt_dir = "./"

    if "resnet" not in classifier_name and not ENSEMBLE:
        classifier = Classifier(input_dim=classifier_input_dim, hidden_dim=classifier_hidden_dim, num_classes=num_classes, seq_len=seq_len, 
                                num_layers=classifier_num_layers, patch_size=patch_size, cell_type=classifier_name, permute_patches=permute_patches).to(device)
        classifier.load_weights(PATH_classifier)
        freeze_parameters(classifier)

        confidnet = Confidnet(input_dim=classifier_hidden_dim, hidden_dim=confidnet_hidden_dim, seq_len=seq_len, num_layers=confidnet_num_layers, batch_first=True).to(device)
        confidnet.load_weights(PATH_confidnet)
        freeze_parameters(confidnet)

        env = WaferMapGrid(classifier, val_loader, test_loader, confidnet=confidnet)
        agent = REINFORCEAgent(num_actions=env.action_space.n, batch_size=patch_selector_batch_size,
                               gamma=gamma, lr=patch_selector_lr, num_iters=patch_selector_num_iters, 
                               input_dims=env.observation_space.shape, seed=SEED,
                               chkpt_dir=chkpt_dir)

        if TRAIN_PATCH_SELECTOR:
            agent = training_loop_reinforce(agent, env, num_iters=patch_selector_num_iters)
        
        agent.load_models()
            
        acc_rl, f1_rl, score, patch_order, stopping = eval_reinforce(agent, env, confidnet=confidnet, 
                                                                     cnn_model=aux_model, supcon=supcon)

        print("Classifier + Confidnet + Patch selector accuracy: ", acc_rl.item())
        print("Classifier + Confidnet + Patch selector F1 score: ", f1_rl.item())
        print("Classifier + Confidnet + Patch selector stopping time: ", stopping.mean().item())
        print("------------------")
        
    # ---------------------------------------------------------------------------- ENSEMBLE EVALUATION ----------------------------------------------------------------------------
    
    if ENSEMBLE and "resnet" not in classifier_name:
        ensemble_type = args["ensemble_type"]
        num_models = args["num_models"]
        
        if HYBRID:
            aux_model_name = args["aux_model_name"]
            PATH_aux_model = f"{aux_model_name}_seed{SEED}.pth"
            if supcon:
                aux_model = SupConResNet(model_name=aux_model_name, num_classes=num_classes, 
                                         input_channels=input_channels,  feat_dim=128, pretrained=False)
            else:
                aux_model = ResNet(model_name=aux_model_name, num_classes=num_classes, 
                                   input_channels=input_channels, pretrained=False)
            
            aux_model = aux_model.to(device)
            aux_model.load_state_dict(torch.load(PATH_aux_model))
        else:
            aux_model = None
        
        classifiers = []
        confidnets = []
        SEEDS = [i for i in range(num_models)]
        
        for SEED in SEEDS:
            classifier = Classifier(input_dim=classifier_input_dim, hidden_dim=classifier_hidden_dim, num_classes=num_classes, seq_len=seq_len, 
                                    num_layers=classifier_num_layers, patch_size=patch_size, cell_type=classifier_name, permute_patches=permute_patches).to(device)
            PATH_classifier = f"{classifier_name}_seed{SEED}.pth"
            classifier.load_weights(PATH_classifier)
            freeze_parameters(classifier)
            classifiers.append(classifier)
            
            confidnet = Confidnet(input_dim=classifier_hidden_dim, hidden_dim=confidnet_hidden_dim, seq_len=seq_len, num_layers=confidnet_num_layers, batch_first=True).to(device)
            PATH_confidnet = f"confid{classifier_name}_seed{SEED}.pth"
            confidnet.load_weights(PATH_confidnet)
            freeze_parameters(confidnet)
            confidnets.append(confidnet)
            
        env = WaferMapGridEnsemble(classifiers, val_loader, test_loader, confidnet=confidnets)
        agent = REINFORCEAgentEnsemble(n_actions=env.action_space.n, gamma=gamma,
                                       input_dims=env.observation_space.shape, ensemble_type=ensemble_type,
                                       num_models=len(SEEDS), seeds=SEEDS, chkpt_dir=chkpt_dir)
        
        agent.load_models()
        
        acc_rl_ens, f1_rl_ens, scor_ens, patch_order_ens, stopping_ens = eval_reinforce(agent, env, confidnet=confidnets, 
                                                                                        cnn_model=aux_model, ensemble=ENSEMBLE, supcon=supcon)
        
        print("ENSEMBLE Classifier + Confidnet + Patch selector accuracy: ", acc_rl_ens.item())
        print("ENSEMBLE Classifier + Confidnet + Patch selector F1 score: ", f1_rl_ens.item())
        print("ENSEMBLE Classifier + Confidnet + Patch selector stopping time: ", stopping_ens.mean().item())
        print("------------------")
            
    
    return 0



if __name__ == "__main__":
    main()