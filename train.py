import random
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import structural_similarity as ssim
from data_loader.MovingMNIST import MovingMNIST
from model.DLinear_Encode2Decode import Encode2Decode
from model.classification.squeezenet import  squeezenet,generator
from torch import optim
from pathlib import Path
#import pytorch_ssim

from utils import save_images


def split_train_val(dataset):
    idx = [i for i in range(len(dataset))]

    random.seed(1234)
    random.shuffle(idx)

    num_train = int(0.8 * len(idx))

    train_idx = idx[:num_train]
    val_idx = idx[num_train:]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    print(f'train index: {len(train_idx)}')
    print(f'val index: {len(val_idx)}')

    return train_dataset, val_dataset


def train(model,generator,discriminator,optimizer_D,optimizer_G, train_dataloader, valid_dataloader, criterion,optimizer_model, device="cpu", epochs=10,
          save_dir=Path("outputs")):
    scheduler_model = ReduceLROnPlateau(optimizer_model, 'min', factor=0.5, patience=5, verbose=True)
    scheduler_G = ReduceLROnPlateau(optimizer_G, 'min', factor=0.5, patience=50, verbose=True)
    scheduler_D = ReduceLROnPlateau(optimizer_D, 'min', factor=0.5, patience=50, verbose=True)
    criterion_BCELoss = nn.BCELoss()
    # ssim_loss = pytorch_ssim.SSIM()
    n_total_steps = len(train_dataloader)
    train_losses = []
    # model.load_state_dict(torch.load(os.path.join(save_dir,'checkpoint_epoch_129.pt')))
    # start_epoch=129
    # print("checkpoint_epoch_129.pt")
    ckpt_path = os.path.join(save_dir, f'checkpoint_epochs_{epochs}.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path,map_location='cpu')
        model.load_state_dict(ckpt['model'])
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        optimizer_G.load_state_dict(ckpt['optimizer_G'])
        scheduler_G.load_state_dict(ckpt['scheduler_G'])
        optimizer_D.load_state_dict(ckpt['optimizer_D'])
        scheduler_D.load_state_dict(ckpt['scheduler_D'])
        optimizer_model.load_state_dict(ckpt['optimizer_model'])
        scheduler_model.load_state_dict(ckpt['scheduler_model'])
        start_epoch = ckpt['epoch']
        start_epoch += 1
    else:
        start_epoch = 0
    avg_train_losses = []
    avg_valid_losses = []
    pre_epoch=500
    for i in range(start_epoch,epochs):
        losses, val_losses = [], []
        # generator.train()
        model.train()
        # discriminator.train()
        epoch_loss = 0.0
        val_epoch_loss = 0.0




        #         index_length = len([n for n,m in model.named_parameters()])
        #         drop = min(index_length//2,i)
        #         index_list = random.sample(range(0,index_length),drop)
        #         index_list.sort()
        #         print(index_list)
        #         temp_k = 0
        #         for m,n in model.named_parameters():
        #             if i>=1:s
        #                 if temp_k in index_list:
        #                     n.requires_grad=False
        #                     print("done",temp_k,m)
        #                 else:
        #                     n.requires_grad=True
        #             temp_k+=1




        #         for m,n in model.named_parameters():
        #             print(m,n)

        # for m,n in model.named_parameters():
        #     print(m)
        for ite, (x, y) in enumerate(train_dataloader):
            x_train, y_train = x.to(device), y.to(device)
            pred_train,pred_train_0,generator,discriminator,optimizer_D,optimizer_G = model(i,pre_epoch,x_train,y_train,generator,discriminator,optimizer_D,optimizer_G)


            # optimizer_D.zero_grad()
            # D_Y = discriminator(y_train)
            # D_P_D = discriminator(pred_train_D)
            # D_Y = torch.std
            # print(D_Y)
            # loss_D = D_P_D-D_Y
            # loss_D.backward()
            # optimizer_D.step()
            # train_losses_D.append(loss_D.item())


            # optimizer_G.zero_grad()
            # pred_train_G = generator(x_train, y_train)
            # for m,n in discriminator.named_parameters():
            #     n.requires_grad=False
            # D_Y = discriminator(y_train)
            # D_P_G = torch.std(discriminator(pred_train_G))
            # loss_G = criterion(pred_train_G,y_train)
            # if i>=1:
            #     loss_G += criterion(D_Y,D_P_G)
            # loss_G = criterion(pred_train_G,y_train) + ((i+1)/epochs+0.5)*criterion(D_Y,D_P_G)
            # loss_G = -2*D_P_G
            # loss_G.backward()
            # optimizer_G.step()
            # epoch_loss_G += loss_G.item()
            # train_losses_G.append(loss_G.item())
            # for m,n in discriminator.named_parameters():
            #     n.requires_grad=True
            with torch.autograd.set_detect_anomaly(True):
                optimizer_model.zero_grad()
                # loss = -ssim_loss(pred_train_0,y_train)
                loss = criterion(pred_train_0,y_train)
                # loss_list = []
                # for m in range(pred_train_0.size(0)):
                #   # for n in range(pred_train_0.size(1)):
                #     loss_list.append(-ssim_loss(pred_train_0[m],y_train[m]))

                # loss = torch.mean(torch.stack(loss_list,dim=0))

                print("loss_0:",loss.item())
                loss.backward()
                optimizer_model.step()
                # loss = criterion(pred_train,y_train)
                # loss_list = []
                # for m in range(pred_train_0.size(0)):
                #   # for n in range(pred_train_0.size(1)):
                #     loss_list.append(-ssim_loss(pred_train_0[m],y_train[m]))
                # if i>=1:
                #   loss = torch.mean(torch.stack(loss_list,dim=0))
                # else:
                loss = criterion(pred_train,y_train)
                train_losses.append(loss.item())


            x_train_0 = x_train
            y_train_0 = y_train
            pred_train_0 = pred_train
            plt.subplot(161)
            plt.imshow(x_train_0.cpu().data.numpy()[0][0][0])
            plt.subplot(162)
            plt.imshow(x_train_0.cpu().data.numpy()[0][9][0])
            plt.subplot(163)
            plt.imshow(y_train_0.cpu().data.numpy()[0][0][0])
            plt.subplot(164)
            plt.imshow(y_train_0.cpu().data.numpy()[0][9][0])
            plt.subplot(165)
            plt.imshow(pred_train_0.cpu().data.numpy()[0][0][0])
            plt.subplot(166)
            plt.imshow(pred_train_0.cpu().data.numpy()[0][9][0])
            plt.savefig(save_dir / f"train_pred_last_{epochs}.png")
            plt.clf()




            print(
                f'epoch {i + 1} / {epochs}, step {ite + 1}/{n_total_steps},loss = {loss.item():.4f}')
            # f = open("epoch_"+str(epochs)+"_loss.txt","a+")
            # f.write(f'G_epoch {i + 1} / {epochs}, step {ite + 1}/{n_total_steps}, loss = {loss_G.item():.4f} \n')
            # f.close()
            # print(f'D_loss = {loss_D.item():.4f}')
            f = open("epoch_"+str(epochs)+"_loss.txt","a+")
            f.write(f'epoch {i + 1} / {epochs}, step {ite + 1}/{n_total_steps}, loss = {loss.item():.4f} \n')
            f.close()



        with torch.no_grad():
            model.eval()
            for _, (x, y) in enumerate(valid_dataloader):
                x_val, y_val = x.to(device), y.to(device)
                pred_val,pred_val_0,generator,discriminator,optimizer_D,optimizer_G = model(i,pre_epoch,x_val, y_val,generator,discriminator,optimizer_D,optimizer_G, teacher_forcing_rate=0,val_flag=True)
                loss = criterion(pred_val, y_val)
                val_losses.append(loss.item())
                val_epoch_loss += loss.item()
        x_train_0 = x_val
        y_train_0 = y_val
        pred_train_0 = pred_val
        plt.subplot(161)
        plt.imshow(x_train_0.cpu().data.numpy()[0][0][0])
        plt.subplot(162)
        plt.imshow(x_train_0.cpu().data.numpy()[0][9][0])
        plt.subplot(163)
        plt.imshow(y_train_0.cpu().data.numpy()[0][0][0])
        plt.subplot(164)
        plt.imshow(y_train_0.cpu().data.numpy()[0][9][0])
        plt.subplot(165)
        plt.imshow(pred_train_0.cpu().data.numpy()[0][0][0])
        plt.subplot(166)
        plt.imshow(pred_train_0.cpu().data.numpy()[0][9][0])
        plt.savefig(save_dir / f"val_pred_last_{epochs}.png")
        plt.clf()

        train_loss = np.average(train_losses)
        # train_loss_G = np.average(train_losses_G)
        valid_loss = np.average(val_losses)
        scheduler_model.step(train_loss)
        scheduler_D.step(train_loss)
        scheduler_G.step(train_loss)
        # print("G",optimizer_G.state_dict()['param_groups'][0]['lr'])
        # train_loss_D = np.average(train_losses_D)
        # scheduler_D.step(train_loss_D)
        # print("D:",optimizer_D.state_dict()['param_groups'][0]['lr'])
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        state = {
            'model': model.state_dict(),
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'scheduler_G': scheduler_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'scheduler_D': scheduler_D.state_dict(),
            'optimizer_model': optimizer_model.state_dict(),
            'scheduler_model': scheduler_model.state_dict(),
            'epoch': i,
        }
        torch.save(state, save_dir / f"checkpoint_epochs_{epochs}.pt")
        print('{}th learning rate {},train loss {}, valid loss {}'.format(i, optimizer_model.state_dict()['param_groups'][0]['lr'], np.mean(train_loss), np.mean(valid_loss)))
        f = open("epoch_"+str(epochs)+"_loss_result.txt","a+")
        f.write('{}th epochs learning rate {},train loss {}, valid loss {} \n'.format(i, optimizer_model.state_dict()['param_groups'][0]['lr'], np.mean(train_loss), np.mean(valid_loss)))
        f.close()

        os.system(f"git add outputs/train_pred_last_{epochs}.png")
        os.system(f"git add outputs/val_pred_last_{epochs}.png")
        os.system(f"git add outputs/checkpoint_epochs_{epochs}.pt")
        os.system(f"git add epoch_{epochs}_loss_result.txt")
        os.system(f"git add epoch_{epochs}_loss.txt")
        os.system("git config --global user.email \"2358384171@qq.com\"")
        os.system("git config --global user.name \"17683995446\"")
        os.system(f'git commit -m \"{epochs},{i},{ite}\" ')
        os.system("git push --force origin HEAD:0917")


        plt.plot(avg_train_losses, '-o')
        plt.plot(avg_valid_losses, '-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train', 'Validation'])
        plt.title('(MSE) Avg Train vs Validation Losses')
        plt.savefig(save_dir / f"train_val_loss_curve_epoch_{i}.png")
        plt.clf()


def test(model, test_dataloader, criterion, ckp_path, device):
    model.load_state_dict(torch.load(ckp_path, map_location=device))
    test_pred, test_gt = [], []
    with torch.no_grad():
        model.eval()
        for i, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x, y, teacher_forcing_rate=0)
            test_pred.append(pred.cpu().data.numpy())
            test_gt.append(y.cpu().data.numpy())
    test_pred = np.concatenate(test_pred)
    test_gt = np.concatenate(test_gt)
    mse = criterion(test_gt, test_pred)
    print('TEST Data loader - MSE = {:.6f}'.format(mse))

    # Frame-wise comparison in MSE and SSIM

    overall_mse = 0
    overall_ssim = 0
    frame_mse = np.zeros(test_gt.shape[1])
    frame_ssim = np.zeros(test_gt.shape[1])

    for i in range(test_gt.shape[1]):
        for j in range(test_gt.shape[0]):
            mse_ = np.square(test_gt[j, i] - test_pred[j, i]).sum()
            test_gt_img = np.squeeze(test_gt[j, i])
            test_pred_img = np.squeeze(test_pred[j, i])
            ssim_ = ssim(test_gt_img, test_pred_img)

            overall_mse += mse_
            overall_ssim += ssim_
            frame_mse[i] += mse_
            frame_ssim[i] += ssim_

    overall_mse /= 10
    overall_ssim /= 10
    frame_mse /= 1000
    frame_ssim /= 1000
    print(f'overall_mse.shape {overall_mse}')
    print(f'overall_ssim.shape {overall_ssim}')
    print(f'frame_mse.shape {frame_mse}')
    print(f'frame_ssim.shape {frame_ssim}')

    path_pred = './results/npy_file_save/saconvlstm_test_pred_speedpt5.npy'
    path_gt = './results/npy_file_save/saconvlstm_test_gt_speedpt5.npy'

    np.save(path_pred, test_pred)
    np.save(path_gt, test_gt)


def get_config():
    # TODO: get config from yaml file
    config = {
        "epoch": 9170,
        'input_dim': 1,
        'batch_size': 32,
        'padding': 1,
        'lr': 0.001,
        'device': "cuda:0" if torch.cuda.is_available() else "cup",
        'attn_hidden_dim': 64,
        'kernel_size': (3, 3),
        'img_size': (16, 16),
        'hidden_dim': 64,
        'num_layers': 1,
        'output_dim': 10,
        'input_window_size': 10,
        'loss': "L2",
        'model_cell': 'sa_convlstm',
        'bias': True,
        "batch_first": True,
        "root": 'data_loader/.data/mnist'
    }
    return config


def main():
    config = get_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = MovingMNIST(config["root"], train=True,download=True)
    train_dataset, val_dataset = split_train_val(dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                                   num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                                   num_workers=4)
    # test_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
    #                                               num_workers=0)
    # ***************** loss function  *********************#
    criterion = nn.MSELoss()
    # ***************** model ******************************#
    from model.classification.squeezenet import generator
    generator = generator(class_num=1)
    discriminator = squeezenet(class_num=1)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=config['lr'])
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'])

    model = Encode2Decode(config['input_dim'],
                          config['hidden_dim'],
                          attn_hidden_dim=config["attn_hidden_dim"],
                          kernel_size=config['kernel_size'],
                          img_size=config["img_size"],
                          num_layers=config['num_layers'],
                          batch_first=config['batch_first'],
                          bias=config['bias']
                          )
    model= model.to(device)
    # ******************  optimizer ***********************#
    optimizer_model = optim.Adam(model.parameters(), lr=config['lr'])
    # ****************** start training ************************#
    train(model,generator,discriminator,optimizer_D,optimizer_G, train_dataloader, valid_dataloader, criterion,optimizer_model,device,epochs=config["epoch"])
    # print(next(iter(train_dataloader))[0].shape)
    # test_data = MovingMNIST(config["root"], train=False)
    # test(model, test_dataloader, criterion,ckp_path=,device)


if __name__ == '__main__':
    main()