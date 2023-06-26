import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
Define task metrics, loss functions and model trainer here.
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss

class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


"""
=========== Universal Multi-task Trainer =========== 
"""


def multi_net_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt, n_class=13):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([opt.epochs, 24], dtype=np.float32)
    print("starting training")
    for index in range(opt.epochs):
        loop = tqdm(train_loader)
        cost = np.zeros(24, dtype=np.float32)
        # iteration for all batches
        multi_task_model.train()
        conf_mat = ConfMatrix(n_class)
        for train_data, train_label, train_depth, train_normal in loop:

            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred = multi_task_model(train_data)
            # print('output predicted')

            optimizer.zero_grad()
            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth'),
                          model_fit(train_pred[2], train_normal, 'normal')]

            # print('loss calculated')

            loss = sum(train_loss[i] for i in range(3))

            loss.backward()
            optimizer.step()

            # print('backprop done')

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch

            loop.set_description(f"Epoch [{index}/{opt.epochs}]")
            loop.set_postfix(loss=loss.item())
            # print('cost calculated')

        # compute mIoU and acc
        avg_cost[index, 1:3] = np.array(conf_mat.get_metrics())

        # evaluating test data
        print("satrting validation")
        multi_task_model.eval()
        conf_mat = ConfMatrix(n_class)
        with torch.no_grad():  # operations inside don't track history
            loop = tqdm(test_loader)
            for test_data, test_label, test_depth, test_normal in loop:

                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth'),
                             model_fit(test_pred[2], test_normal, 'normal')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
                avg_cost[index, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 13:15] = np.array(conf_mat.get_metrics())

        scheduler.step()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                    avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                    avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))

        if index%5==0:
            state_dict = {"model_state_dict":multi_task_model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": index, "loss": avg_cost}
            torch.save(state_dict, os.path.join(opt.ckpt_dir, f"model_epoch_{index}.pth"))
"""
=========== Universal Single-task Trainer =========== 
"""


def single_task_trainer(train_loader, test_loader, single_task_model, device, optimizer, scheduler, task, ckpt_dir, total_epoch=200, n_class=13):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    for index in range(total_epoch):
        cost = np.zeros(24, dtype=np.float32)
        loop = tqdm(train_loader)
        # iteration for all batches
        single_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(n_class)
        for  train_data, train_label, train_depth, train_normal in loop:
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred = single_task_model(train_data)
            optimizer.zero_grad()

            if task == 'semantic':
                train_loss = model_fit(train_pred, train_label, task)
                train_loss.backward()
                optimizer.step()

                conf_mat.update(train_pred.argmax(1).flatten(), train_label.flatten())
                cost[0] = train_loss.item()

            if task == 'depth':
                train_loss = model_fit(train_pred, train_depth, task)
                train_loss.backward()
                optimizer.step()
                cost[3] = train_loss.item()
                cost[4], cost[5] = depth_error(train_pred, train_depth)

            if task == 'normal':
                train_loss = model_fit(train_pred, train_normal, task)
                train_loss.backward()
                optimizer.step()
                cost[6] = train_loss.item()
                cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred, train_normal)

            avg_cost[index, :12] += cost[:12] / train_batch
            loop.set_description(f"Epoch [{index}/{total_epoch}]")
            loop.set_postfix(loss=train_loss.item())

        if task == 'semantic':
            avg_cost[index, 1:3] = np.array(conf_mat.get_metrics())

        # evaluating test data
        print('starting validation')
        single_task_model.eval()
        conf_mat = ConfMatrix(n_class)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            loop = tqdm(test_loader)
            for test_data, test_label, test_depth, test_normal in loop:
                test_data, test_label = test_data.to(device),  test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = single_task_model(test_data)

                if task == 'semantic':
                    test_loss = model_fit(test_pred, test_label, task)

                    conf_mat.update(test_pred.argmax(1).flatten(), test_label.flatten())
                    cost[12] = test_loss.item()

                if task == 'depth':
                    
                    test_loss = model_fit(test_pred, test_depth, task)
                    cost[15] = test_loss.item()
                    cost[16], cost[17] = depth_error(test_pred, test_depth)
                    print(cost[16], cost[17])
                    test_pred = test_pred.detach().cpu()
                    test_depth = test_depth.cpu()
                    plt.imshow(test_depth[0][0])
                    plt.show()
                    plt.imshow(test_pred[0][0])
                    plt.show()

                if task == 'normal':
                    test_loss = model_fit(test_pred, test_normal, task)
                    cost[18] = test_loss.item()
                    cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred, test_normal)

                avg_cost[index, 12:] += cost[12:] / test_batch
            if task == 'semantic':
                avg_cost[index, 13:15] = np.array(conf_mat.get_metrics())

        scheduler.step()
        if task == 'semantic':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 12], avg_cost[index, 13], avg_cost[index, 14]))
        if task == 'depth':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17]))
        if task == 'normal':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11],
                      avg_cost[index, 18], avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))

        
        if index%5==0:
            state_dict = {"model_state_dict":single_task_model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": index, "loss": avg_cost}
            torch.save(state_dict, os.path.join(ckpt_dir, f"model_epoch_{index}.pth"))