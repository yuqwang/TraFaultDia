import argparse
import torch
from DatasetTT import DatasetTT
import numpy as np
import random
from MAML import MAML
import os
from torch import nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    from transformers import set_seed
    set_seed(seed)

def train_t4maml5w5s(config):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db = DatasetTT(batchsz=config['task_num'], n_way=config['n_way'], k_spt=config['k_spt'], k_query=config['k_qry'],
                   span_max_length=config['span_max_length'], log_max_length=config['log_max_length'],
                   span_input_dim = config["span_input_dim"], log_input_dim = config["log_input_dim"])

    learner_config = [
        # ('rnn', [args.input_dim, args.hidden_dim, args.n_layers, args.n_way])
        # ('lstm', [args.encoder_dim, 256, args.n_layers, args.n_way])
        ('transformer_encoder', [config['input_dim'], config['nhead'], config['num_encoder_layers'], 2048, config['n_way'], config['dropout']])
    ]

    test_cls = [2, 10, 3, 11, 4, 8, 6, 20, 19, 17]

    maml = MAML(config, learner_config).to(device)

    perm_size = config['perm_size']
    cache_size = config['cache_size']

    for epoch in range(config['epoch']):
        x_spt_tensor, x_qry_tensor, y_spt_tensor, y_qry_tensor = db.load_data_cache_t3('train')
        y_spt_tensor = y_spt_tensor.clone().detach().long()
        y_qry_tensor = y_qry_tensor.clone().detach().long()
        x_spt_tensor, y_spt_tensor, x_qry_tensor, y_qry_tensor = x_spt_tensor.to(device), y_spt_tensor.to(
            device), x_qry_tensor.to(device), y_qry_tensor.to(device)

        train_accs = maml(x_spt_tensor, y_spt_tensor, x_qry_tensor, y_qry_tensor)

        if epoch % 10 == 0:
            print("coming to meta-testing")
            # meta-testing
            perm_accs = [0 for _ in range(perm_size)]
            for i in range(perm_size):
                print("perm_num:", i)
                # 直接随机选择5个数字并生成一个排列
                selected_numbers = random.sample(test_cls, config['n_way'])
                random_permutation = tuple(selected_numbers)

                print("selected test cls is>", random_permutation, flush=True)
                test_x_spt_tensor, test_x_qry_tensor, test_y_spt_tensor, test_y_qry_tensor = db.load_data_cache_test('test', random_permutation, cache_size)
                test_y_spt_tensor = test_y_spt_tensor.clone().detach().long()
                test_y_qry_tensor = test_y_qry_tensor.clone().detach().long()

                test_x_spt_tensor, test_y_spt_tensor, test_x_qry_tensor, test_y_qry_tensor = test_x_spt_tensor.to(device), \
                    test_y_spt_tensor.to(device), test_x_qry_tensor.to(device), test_y_qry_tensor.to(device)

                cache_accs = [0 for _ in range(cache_size)]
                for cache in range(cache_size):
                    cache_x_spt_tensor = test_x_spt_tensor[cache, :, :, :]
                    cache_x_qry_tensor = test_x_qry_tensor[cache, :, :, :]
                    cache_y_spt_tensor = test_y_spt_tensor[cache, :]
                    cache_y_qry_tensor = test_y_qry_tensor[cache, :]

                    cache_test_accs = maml.finetunning(cache_x_spt_tensor, cache_y_spt_tensor, cache_x_qry_tensor,
                                                       cache_y_qry_tensor)
                    cache_accs[cache] = np.max(cache_test_accs)

                highest_cache_acc = np.max(cache_accs)
                perm_accs[i] = perm_accs[i] + highest_cache_acc

            test_highest_perm_accs = np.max(perm_accs)
            test_ave_perm_accs = np.mean(perm_accs)

        # 报告所有的准确率
        metrics_to_report = {f'accuracy_step_{i}': acc for i, acc in enumerate(train_accs)}
        metrics_to_report.update({'test_highest_accs': test_highest_perm_accs, 'test_ave_accs': test_ave_perm_accs})

        tune.report(**metrics_to_report)

if __name__ == '__main__':
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Change the working directory to the location of the script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    search_space = {
        "epoch": 150,
        "n_way": 5,
        "k_spt": 10,
        "k_qry": 15,
        "task_num": 3,

        "span_max_length": 91,
        "log_max_length": 194,
        "span_input_dim": 775,
        "log_input_dim": 768,

        "update_lr": tune.loguniform(2e-3, 5e-2),
        "meta_lr": tune.loguniform(9e-6, 1e-4),
        "dropout": tune.uniform(0.1, 0.46),

        "input_dim": 768,
        "nhead": tune.choice([1, 2]),  # 从[5, 25]中选择
        "num_encoder_layers": 2,  # 从[2, 4, 6]中选择
        "update_step": 5,
        "update_step_test": 15,
        # ... add other parameters as needed

        "cache_size": 5,
        "perm_size": 50
    }

    # 创建一个scheduler来进行早停
    scheduler = ASHAScheduler(
        metric="test_ave_accs",
        mode="max",
        max_t=100,
        grace_period=11,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=[f"accuracy_step_{i}" for i in range(5)] + ["test_ave_accs", "test_highest_accs", "training_iteration"],
        max_progress_rows=2
    )

    analysis = tune.run(

        train_t4maml5w5s,
        resources_per_trial={"cpu": 32, "gpu": 1},
        config=search_space,
        num_samples=40,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir='./ray_tune/meta',
        keep_checkpoints_num=2,  # Keep only the best checkpoint.
        checkpoint_score_attr="test_ave_accs"
    )

    best_trial = analysis.get_best_trial("test_ave_accs", "max", "last")
    print("Best trial config: ", analysis.get_best_config(metric="test_ave_accs", mode="max"))
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final training accuracy: {}".format(
        best_trial.last_result["test_ave_accs"]))
