from Learner import Learner
import torch
from torch import optim, nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import time


class MAML(nn.Module):
    """
        Meta Learner
        """

    def __init__(self, config, learner_config):
        '''
           Initialize the MAML model with configuration parameters for both the meta-learning process
           and the learner configuration for task-specific adaptation.

           Parameters:
           - config: A dictionary containing configuration parameters for the meta-learning process,
           - learner_config: A dictionary containing configuration parameters for meta-learner
           '''

        super(MAML, self).__init__()

        # Learning rate for task-level updates during adaptation.
        self.update_lr = config["update_lr"]
        # Learning rate for updates to meta-parameters
        self.meta_lr = config["meta_lr"]
        self.n_way = config["n_way"]
        self.k_spt = config["k_spt"]
        self.k_qry = config["k_qry"]
        # The number of meta-training tasks
        self.task_num = config["task_num"]
        # The number of gradient updates to be performed on each task in the support set during adaptation.
        self.update_step = config["update_step"]
        # The number of gradient updates to be performed on each task in the query set during testing.
        self.update_step_test = config["update_step_test"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Learner(learner_config).to(device)
        self.meta_optim = optim.AdamW(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
         setsz for a task = k_spt * n_way
         querysz for a teask = k_query * n_way
                :param x_spt:   [Task_num, setsz, max_tokens, embedding_length]
                :param y_spt:   [Task_num, setsz]
                :param x_qry:   [Task_num, querysz, max_tokens, embedding_length]
                :param y_qry:   [Task_num, querysz]
                :return:

                temp_model = copy.deepcopy(self.net)
                """
        task_num, setsz, max_tokens, embedding_length = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        network_params = []

        for i in range(task_num):
            if i == 0:
                #print("step 0")
                logits = self.net(x_spt[i])
                net_params = list(dict(self.net.named_parameters()).values())  # Extract parameter tensors and convert to list
                network_params = net_params
            else:
                #print("step 1")
                net_params = network_params
                logits = self.net.updated_forward(x_spt[i],net_params)

            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i])
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net.updated_forward(x_qry[i], fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i])
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, self.net.parameters())
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net.updated_forward(x_qry[i], fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()

        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs  # loss_q, pred

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
                :param x_spt:   [setsz, max_tokens, embedding_length]
                :param y_spt:   [setsz]
                :param x_qry:   [querysz, max_tokens, embedding_length]
                :param y_qry:   [querysz]
                :return:
                """
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        outloop_net = deepcopy(self.net)

        adaptation_start_time = time.time()
        # 1. run the i-th task and compute loss for k=0
        y_hat = outloop_net(x_spt)
        loss = F.cross_entropy(y_hat, y_spt)
        grad = torch.autograd.grad(loss, outloop_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, outloop_net.parameters())))

        # the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = outloop_net(x_qry)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] += correct

        # the loss and accuracy after the first update
        with torch.no_grad():
            logits_q = outloop_net.updated_forward(x_qry, fast_weights)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        adaptation_end_time = time.time()
        adaptation_elapsed_time = adaptation_end_time - adaptation_start_time  # 计算用时
        print(f"Adaptation time: {adaptation_elapsed_time} seconds")  # 打印用时

        for k in range(1, self.update_step_test):
            logits = outloop_net(x_spt)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, outloop_net.parameters())
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = outloop_net.updated_forward(x_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k+1] += correct

        del outloop_net

        accs = np.array(corrects) / querysz

        return accs
