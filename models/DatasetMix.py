import argparse
import copy
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from transformers import BertModel, BertTokenizer, BertConfig
import os
from torch import nn
import random
import itertools
from AttenAE import AttenAEFusionModel
from sklearn.preprocessing import OneHotEncoder
from MAML import MAML

'''
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    from transformers import set_seed
    set_seed(seed)'''


class DatasetMix(Dataset):

    def __init__(self, batchsz, n_way, k_spt, k_query, TTspan_max_length,
                 TTlog_max_length, TTspan_input_dim, TTlog_input_dim,
                 BTspan_max_length, BTlog_max_length, BTspan_input_dim, BTlog_input_dim):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        self.n_task = batchsz  # batch of set args.task_num #
        self.n_way = n_way  # n-way
        self.k_spt = k_spt  # k-shot
        self.k_qry = k_query  # for evaluation
        self.k_sample = self.k_spt + self.k_qry
        self.setsz = self.n_way * self.k_spt  # num of samples per set
        self.querysz = self.n_way * self.k_qry  # number of samples per set for evaluation
        self.total_task_num = self.n_task + 1

        self.TTspan_max_length = TTspan_max_length
        self.TTlog_max_length = TTlog_max_length
        self.TTspan_input_dim = TTspan_input_dim
        self.TTlog_input_dim = TTlog_input_dim

        self.BTspan_max_length = BTspan_max_length
        self.BTlog_max_length = BTlog_max_length
        self.BTspan_input_dim = BTspan_input_dim
        self.BTlog_input_dim = BTlog_input_dim

        self.pretrain_dim = 768
        #OnlineBoutique base categories
        self.train_cls = [1, 4, 5, 6, 10, 12, 13, 14, 15, 18, 19 , 20, 21, 22, 23, 24, 25, 26, 28, 31, 32]

        # AttenAE's optimized encoder for Trainticket

        TT_config = {
            "dropout_rate": 0.14857704629135002,
            "embed_dim": 768,
            "num_heads": 8,
            "span_input_dim": self.TTspan_input_dim,
            "log_input_dim": self.TTlog_input_dim
        }

        # AttenAE's optimized encoder for OnlineBoutique
        OB_config = {
            "dropout_rate": 0.2846091710354427,
            "embed_dim": 768,
            "num_heads": 8,
            "span_input_dim": self.BTspan_input_dim ,
            "log_input_dim": self.BTlog_input_dim
        }

        bert_dir_path = '../local_bert_uncased/'
        self.Bert_model = BertModel.from_pretrained(bert_dir_path, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir_path)

        TTFusion_model_state_dict_path = './TTfusionModels/AEGatedL_best_model.pth'
        BTFusion_model_state_dict_path = '../Boutique/TTfusionModels/AEGatedL_best_model.pth'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.TTFusion_model = AttenAEFusionModel(TT_config)
        #TTFusion_model_state_dict = torch.load(TTFusion_model_state_dict_path, map_location=torch.device('cpu'))
        TTFusion_model_state_dict = torch.load(TTFusion_model_state_dict_path)
        TTFusion_model_state_dict_formatted = {k.replace('module.', ''): v for k, v in TTFusion_model_state_dict.items()}
        self.TTFusion_model.load_state_dict(TTFusion_model_state_dict_formatted)

        self.BTFusion_model = AttenAEFusionModel(OB_config)
        #BTFusion_model_state_dict = torch.load(BTFusion_model_state_dict_path, map_location=torch.device('cpu'))
        BTFusion_model_state_dict = torch.load(BTFusion_model_state_dict_path)
        BTFusion_model_state_dict_formatted = {k.replace('module.', ''): v for k, v in
                                               BTFusion_model_state_dict.items()}
        self.BTFusion_model.load_state_dict(BTFusion_model_state_dict_formatted)

        self.Bert_model = self.Bert_model.to(self.device)
        self.TTFusion_model = self.TTFusion_model.to(self.device)
        self.BTFusion_model = self.BTFusion_model.to(self.device)

        if torch.cuda.device_count() > 1:
            #print("Using", torch.cuda.device_count(), "GPUs!")
            self.Bert_model = nn.DataParallel(self.Bert_model)
            self.TTFusion_model = nn.DataParallel(self.TTFusion_model)
            self.BTFusion_model = nn.DataParallel(self.BTFusion_model)

    def load_data_cache_ob_train(self, datamodel='train'):
        data_cache = []
        # cache_size = 1
        if datamodel == 'train':
            x_spt_tensor = torch.zeros(self.n_task, self.setsz, self.BTlog_max_length, self.pretrain_dim)
            y_spt_tensor = torch.zeros(self.n_task, self.setsz)
            x_qry_tensor = torch.zeros(self.n_task, self.querysz, self.BTlog_max_length, self.pretrain_dim)
            y_qry_tensor = torch.zeros(self.n_task, self.querysz)

            for task_num in range(self.n_task):
                # print("Task number is", task_num)
                total_train_cls = np.random.choice(self.train_cls, self.n_way * self.n_task, False)
                train_cls = total_train_cls[task_num * self.n_way:(task_num + 1) * self.n_way]

                spans_sample_df = pd.DataFrame()
                logs_sample_df = pd.DataFrame()
                x_spt_traces = []
                x_qry_traces = []
                # x_spt_span, x_spt_log, y_spt, x_qry_span, x_qry_log, y_qry = [], [], [], [], [], []
                y_spt_batch_list = []
                y_qry_batch_list = []
                for cls_index, cls_num in enumerate(train_cls):
                    # print("class index is", cls_index, "class number is", cls_num)
                    formatted_file_num = str(cls_num).zfill(2)
                    spans_df = pd.read_csv("../Boutique/data/clean_spans/spanF{}.csv".format(formatted_file_num))
                    logs_df = pd.read_csv("../Boutique/data/clean_logs/logF{}.csv".format(formatted_file_num))

                    spans_df = spans_df.groupby('TraceID').filter(lambda x: len(x) >= 2)
                    unique_trace_Id = spans_df['TraceID'].unique()

                    selected_traces = np.random.choice(unique_trace_Id, size=self.k_sample, replace=False)
                    selected_sample_df = spans_df[spans_df['TraceID'].isin(selected_traces)]
                    spans_sample_df = spans_sample_df._append(selected_sample_df, ignore_index=True)
                    selected_logs_sample_df = logs_df[logs_df['TraceID'].isin(selected_traces)]
                    logs_sample_df = logs_sample_df._append(selected_logs_sample_df, ignore_index=True)
                    x_spt_traces.extend(selected_traces[:self.k_spt])
                    x_qry_traces.extend(selected_traces[self.k_spt:])
                    y_spt_batch_list.extend([cls_index for _ in range(self.k_spt)])
                    y_qry_batch_list.extend([cls_index for _ in range(self.k_qry)])

                spans_sample_df = spans_sample_df.groupby('TraceID', group_keys=False).nth(
                    list(range(self.BTspan_max_length)))
                spans_sample_df = spans_sample_df.reset_index()
                unique_trace_Id1 = spans_sample_df['TraceID'].unique()

                train_span_operation = spans_sample_df['Node_operation'].tolist()
                train_spans_emb = self.create_bert_sentence_emb(train_span_operation, 'BTspanurl')
                unique_trace_Id2 = logs_sample_df['TraceID'].unique()
                logs_sample_df = logs_sample_df.groupby('TraceID', group_keys=False).nth(
                    list(range(self.BTlog_max_length)))
                logs_sample_df = logs_sample_df.reset_index()
                train_logMsgs = logs_sample_df['LogMsgFull'].tolist()
                train_logs_emb = self.create_bert_sentence_emb(train_logMsgs, 'BTlog')

                x_spt_spans_sets_tensor = torch.zeros(self.setsz, self.BTlog_max_length, self.BTspan_input_dim)
                x_spt_logs_sets_tensor = torch.zeros(self.setsz, self.BTlog_max_length, self.BTlog_input_dim)
                x_qry_spans_qrys_tensor = torch.zeros(self.querysz, self.BTlog_max_length, self.BTspan_input_dim)
                x_qry_logs_qrys_tensor = torch.zeros(self.querysz, self.BTlog_max_length, self.BTlog_input_dim)

                # 支持集处理
                for index, trace_id in enumerate(x_spt_traces):
                    x_spt_spans_sets_tensor, x_spt_logs_sets_tensor = self.process_trace(
                        index, trace_id, spans_sample_df, logs_sample_df, train_spans_emb, train_logs_emb,
                        x_spt_spans_sets_tensor, x_spt_logs_sets_tensor)

                # 查询集处理
                for index, trace_id in enumerate(x_qry_traces):
                    x_qry_spans_sets_tensor, x_qry_logs_sets_tensor = self.process_trace(
                        index, trace_id, spans_sample_df, logs_sample_df, train_spans_emb, train_logs_emb,
                        x_qry_spans_qrys_tensor, x_qry_logs_qrys_tensor)

                self.BTFusion_model.eval()

                with torch.no_grad():
                    x_spt_spans_sets_tensor = x_spt_spans_sets_tensor.to(self.device)
                    x_spt_logs_sets_tensor = x_spt_logs_sets_tensor.to(self.device)
                    x_qry_spans_sets_tensor = x_qry_spans_sets_tensor.to(self.device)
                    x_qry_logs_sets_tensor = x_qry_logs_sets_tensor.to(self.device)

                    x_spt_recon_span, x_spt_recon_log, x_spt_fused_tensor = self.BTFusion_model(x_spt_spans_sets_tensor,
                                                                                                x_spt_logs_sets_tensor)
                    x_qry_recon_span, x_qry_recon_log, x_qry_fused_tensor = self.BTFusion_model(x_qry_spans_sets_tensor,
                                                                                                x_qry_logs_sets_tensor)

                # Generate a permutation of indices
                spt_perm = torch.randperm(x_spt_fused_tensor.size(0))
                shuffled_x_spt_fused_tensor = x_spt_fused_tensor[spt_perm]
                y_spt_tem_tensor = torch.tensor(y_spt_batch_list).type(torch.LongTensor)
                shuffled_y_spt_tensor = y_spt_tem_tensor[spt_perm]

                qry_perm = torch.randperm(x_qry_fused_tensor.size(0))
                shuffled_x_qry_fused_tensor = x_qry_fused_tensor[qry_perm]
                y_qry_tem_tensor = torch.tensor(y_qry_batch_list).type(torch.LongTensor)
                shuffled_y_qry_tensor = y_qry_tem_tensor[qry_perm]

                x_spt_tensor[task_num, :, :, :] = shuffled_x_spt_fused_tensor
                y_spt_tensor[task_num, :] = shuffled_y_spt_tensor
                x_qry_tensor[task_num, :, :, :] = shuffled_x_qry_fused_tensor
                y_qry_tensor[task_num, :] = shuffled_y_qry_tensor

        return x_spt_tensor, x_qry_tensor, y_spt_tensor, y_qry_tensor

    def load_data_cache_tt_test(self, datamodel='test', perm=[], cache_size=0):
        data_cache = []
        # cache_size = 1
        if datamodel == 'test':
            test_cls = perm
            cache_size = cache_size
            x_spt_tensor = torch.zeros(cache_size, self.setsz, self.TTlog_max_length, self.pretrain_dim)
            y_spt_tensor = torch.zeros(cache_size, self.setsz)
            x_qry_tensor = torch.zeros(cache_size, self.querysz, self.TTlog_max_length, self.pretrain_dim)
            y_qry_tensor = torch.zeros(cache_size, self.querysz)

            for cache in range(cache_size):
                spans_sample_df_tt = pd.DataFrame()
                logs_sample_df_tt = pd.DataFrame()
                x_spt_traces_tt = []
                x_qry_traces_tt = []
                spans_sample_df_nezha = pd.DataFrame()
                logs_sample_df_nezha = pd.DataFrame()
                x_spt_traces_nezha = []
                x_qry_traces_nezha = []
                y_spt_batch_list_tt = []
                y_qry_batch_list_tt = []
                y_spt_batch_list_nezha = []
                y_qry_batch_list_nezha = []

                for cls_index, cls_num in enumerate(test_cls):
                    # print("class index is", cls_index, "class number is", cls_num)
                    formatted_file_num = str(cls_num).zfill(2)
                    spans_df = pd.read_csv("./Data/span_formatted/SpanF{}.csv".format(formatted_file_num))
                    logs_df = pd.read_csv("./Data/log_clean/F{}logall_clean.csv".format(formatted_file_num))

                    if cls_num < 15:
                        # print("TT datasets")
                        logs_df['SpanIdFull'] = logs_df['SpanIdFull'].str.replace(']]', '', regex=False)
                        selected_sample_df = []
                        # spans_df = spans_df.groupby('TraceId').filter(lambda x: len(x) > 2)
                        unique_trace_Id = spans_df['TraceID'].unique()

                        selected_traces = np.random.choice(unique_trace_Id, size=self.k_sample, replace=False)
                        selected_sample_df = spans_df[spans_df['TraceID'].isin(selected_traces)]
                        spans_sample_df_tt = spans_sample_df_tt._append(selected_sample_df, ignore_index=True)
                        selected_logs_sample_df_TT = logs_df[logs_df['TraceID'].isin(selected_traces)]
                        logs_sample_df_tt = logs_sample_df_tt._append(selected_logs_sample_df_TT, ignore_index=True)
                        x_spt_traces_tt.extend(selected_traces[:self.k_spt])
                        x_qry_traces_tt.extend(selected_traces[self.k_spt:])
                        y_spt_batch_list_tt.extend([cls_index for _ in range(self.k_spt)])
                        # print("x_spt_traces:", x_spt_traces)
                        # print("y_spt:", y_spt_batch_list)
                        y_qry_batch_list_tt.extend([cls_index for _ in range(self.k_qry)])
                        # print("x_qry_traces:", x_qry_traces)
                        # print("y_qry:", y_qry_batch_list)

                    else:
                        unique_trace_id = spans_df['TraceID'].unique()
                        selected_traces = np.random.choice(unique_trace_id, size=self.k_sample, replace=False)
                        selected_sample_df = spans_df[spans_df['TraceID'].isin(selected_traces)]
                        spans_sample_df_nezha = spans_sample_df_nezha._append(selected_sample_df, ignore_index=True)
                        selected_logs_sample_df = logs_df[logs_df['TraceID'].isin(selected_traces)]
                        logs_sample_df_nezha = logs_sample_df_nezha._append(selected_logs_sample_df, ignore_index=True)
                        x_spt_traces_nezha.extend(selected_traces[:self.k_spt])
                        x_qry_traces_nezha.extend(selected_traces[self.k_spt:])
                        y_spt_batch_list_nezha.extend([cls_index for _ in range(self.k_spt)])
                        y_qry_batch_list_nezha.extend([cls_index for _ in range(self.k_qry)])

                if not spans_sample_df_tt.empty:
                    # TT datasets
                    spans_sample_df_tt = spans_sample_df_tt.groupby('TraceID', group_keys=False).nth(
                        list(range(self.TTspan_max_length)))
                    spans_sample_df_tt = spans_sample_df_tt.reset_index()
                    unique_trace_Id1 = spans_sample_df_tt['TraceID'].unique()
                    spans_sample_df_tt['URL'] = spans_sample_df_tt['URL'].apply(
                        lambda x: self.clean_spanurl(str(x)))
                    train_spanurls = spans_sample_df_tt['URL'].tolist()
                    train_spans_emb_tt = self.create_bert_sentence_emb(train_spanurls, 'TTspanurl')
                    unique_trace_Id2 = logs_sample_df_tt['TraceID'].unique()
                    logs_sample_df_tt = logs_sample_df_tt.groupby('TraceID', group_keys=False).nth(
                        list(range(self.TTlog_max_length)))
                    logs_sample_df_tt = logs_sample_df_tt.reset_index()
                    train_logMsgs_TT = logs_sample_df_tt['LogMsgFull'].tolist()
                    train_logs_emb_TT = self.create_bert_sentence_emb(train_logMsgs_TT, 'TTlog')

                if not spans_sample_df_nezha.empty:
                    # Nezha Datasets
                    spans_sample_df_nezha = spans_sample_df_nezha.groupby('TraceID', group_keys=False).nth(
                        list(range(self.TTspan_max_length)))
                    spans_sample_df_nezha = spans_sample_df_nezha.reset_index()
                    unique_trace_Id1 = spans_sample_df_nezha['TraceID'].unique()

                    train_span_operation = spans_sample_df_nezha['Node_operation'].tolist()
                    train_spans_emb_nezha = self.create_bert_sentence_emb(train_span_operation, 'BTspanurl')
                    unique_trace_Id2 = logs_sample_df_nezha['TraceID'].unique()
                    logs_sample_df_nezha = logs_sample_df_nezha.groupby('TraceID', group_keys=False).nth(
                        list(range(self.TTlog_max_length)))
                    logs_sample_df_nezha = logs_sample_df_nezha.reset_index()
                    train_logMsgs_nezha = logs_sample_df_nezha['LogMsgFull'].tolist()
                    train_logs_emb_nezha = self.create_bert_sentence_emb(train_logMsgs_nezha, 'BTlog')

                # print("train_logs_emb：",train_logs_emb)
                x_spt_spans_sets_tensor = torch.zeros(self.setsz, self.TTlog_max_length, self.TTspan_input_dim)
                x_spt_logs_sets_tensor = torch.zeros(self.setsz, self.TTlog_max_length, self.TTlog_input_dim)
                x_qry_spans_sets_tensor = torch.zeros(self.querysz, self.TTlog_max_length, self.TTspan_input_dim)
                x_qry_logs_sets_tensor = torch.zeros(self.querysz, self.TTlog_max_length, self.TTlog_input_dim)

                if not spans_sample_df_tt.empty:
                    # TT支持集处理
                    for index, trace_id in enumerate(x_spt_traces_tt):
                        x_spt_spans_sets_tensor[:len(x_spt_traces_tt), :, :], x_spt_logs_sets_tensor[
                                                                              :len(x_spt_traces_tt), :,
                                                                              :] = self.process_trace(
                            index, trace_id, spans_sample_df_tt, logs_sample_df_tt, train_spans_emb_tt,
                            train_logs_emb_TT,
                            x_spt_spans_sets_tensor[:len(x_spt_traces_tt), :, :],
                            x_spt_logs_sets_tensor[:len(x_spt_traces_tt), :, :])

                    # TT查询集处理
                    for index, trace_id in enumerate(x_qry_traces_tt):
                        x_qry_spans_sets_tensor[:len(x_qry_traces_tt), :, :], x_qry_logs_sets_tensor[
                                                                              :len(x_qry_traces_tt), :,
                                                                              :] = self.process_trace(
                            index, trace_id, spans_sample_df_tt, logs_sample_df_tt, train_spans_emb_tt,
                            train_logs_emb_TT,
                            x_qry_spans_sets_tensor[:len(x_qry_traces_tt), :, :],
                            x_qry_logs_sets_tensor[:len(x_qry_traces_tt), :, :])

                if not spans_sample_df_nezha.empty:
                    for index, trace_id in enumerate(x_spt_traces_nezha):
                        x_spt_spans_sets_tensor[len(x_spt_traces_tt):, :, :770], x_spt_logs_sets_tensor[
                                                                                 len(x_spt_traces_tt):, :,
                                                                                 :770] = self.process_trace(
                            index, trace_id, spans_sample_df_nezha, logs_sample_df_nezha, train_spans_emb_nezha,
                            train_logs_emb_nezha,
                            x_spt_spans_sets_tensor[len(x_spt_traces_tt):, :, :770],
                            x_spt_logs_sets_tensor[len(x_spt_traces_tt):, :, :770])

                    for index, trace_id in enumerate(x_qry_traces_nezha):
                        x_qry_spans_sets_tensor[len(x_qry_traces_tt):, :, :770], x_qry_logs_sets_tensor[
                                                                                 len(x_qry_traces_tt):, :,
                                                                                 :770] = self.process_trace(
                            index, trace_id, spans_sample_df_nezha, logs_sample_df_nezha, train_spans_emb_nezha,
                            train_logs_emb_nezha,
                            x_qry_spans_sets_tensor[len(x_qry_traces_tt):, :, :770],
                            x_qry_logs_sets_tensor[len(x_qry_traces_tt):, :, :770])

                self.TTFusion_model.eval()

                with torch.no_grad():
                    x_spt_spans_sets_tensor = x_spt_spans_sets_tensor.to(self.device)
                    x_spt_logs_sets_tensor = x_spt_logs_sets_tensor.to(self.device)
                    x_qry_spans_sets_tensor = x_qry_spans_sets_tensor.to(self.device)
                    x_qry_logs_sets_tensor = x_qry_logs_sets_tensor.to(self.device)

                    x_spt_recon_span, x_spt_recon_log, x_spt_fused_tensor = self.TTFusion_model(x_spt_spans_sets_tensor,
                                                                                              x_spt_logs_sets_tensor)
                    x_qry_recon_span, x_qry_recon_log, x_qry_fused_tensor = self.TTFusion_model(x_qry_spans_sets_tensor,
                                                                                              x_qry_logs_sets_tensor)

                # Generate a permutation of indices
                y_spt_batch_list = y_spt_batch_list_tt + y_spt_batch_list_nezha
                y_qry_batch_list = y_qry_batch_list_tt + y_qry_batch_list_nezha

                spt_perm = torch.randperm(x_spt_fused_tensor.size(0))
                shuffled_x_spt_fused_tensor = x_spt_fused_tensor[spt_perm]
                y_spt_tem_tensor = torch.tensor(y_spt_batch_list).type(torch.LongTensor)
                shuffled_y_spt_tensor = y_spt_tem_tensor[spt_perm]

                qry_perm = torch.randperm(x_qry_fused_tensor.size(0))
                shuffled_x_qry_fused_tensor = x_qry_fused_tensor[qry_perm]
                y_qry_tem_tensor = torch.tensor(y_qry_batch_list).type(torch.LongTensor)
                shuffled_y_qry_tensor = y_qry_tem_tensor[qry_perm]

                x_spt_tensor[cache, :, :, :] = shuffled_x_spt_fused_tensor
                y_spt_tensor[cache, :] = shuffled_y_spt_tensor
                x_qry_tensor[cache, :, :, :] = shuffled_x_qry_fused_tensor
                y_qry_tensor[cache, :] = shuffled_y_qry_tensor

        return x_spt_tensor, x_qry_tensor, y_spt_tensor, y_qry_tensor

    def process_trace(self, index, trace_id, spans_sample_df, logs_sample_df, train_spans_emb, train_logs_emb,
                      spans_batch_tensor, logs_batch_tensor):
        trace_spans_df = spans_sample_df[spans_sample_df['TraceID'] == trace_id]

        selected_features = ['Normalized_StartTime', 'Normalized_EndTime', 'Normalized_tree_span_ids']
        feature_values_list = trace_spans_df[selected_features].values.tolist()
        combined_span_sample_tensor = torch.tensor(feature_values_list)

        span_indexes = spans_sample_df[spans_sample_df['TraceID'] == trace_id].index
        trace_spans_emb_tensor = train_spans_emb[span_indexes]
        #combined_span_sample_tensor = combined_span_sample_tensor.to(self.device)
        spans_tensor = torch.cat((trace_spans_emb_tensor, combined_span_sample_tensor), dim=1)
        spans_batch_tensor[index, :spans_tensor.shape[0], :] = spans_tensor

        log_indexes = logs_sample_df[logs_sample_df['TraceID'] == trace_id].index
        if not log_indexes.empty:
            trace_logMsgs_tensor = train_logs_emb[log_indexes]
            logs_batch_tensor[index, :trace_logMsgs_tensor.shape[0], :] = trace_logMsgs_tensor

        return spans_batch_tensor, logs_batch_tensor

    def clean_spanurl(self, sentence):
        cleaned_sentence = sentence.lower()
        cleaned_sentence = re.sub(r'\b[0-9a-f]{8}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{12}\b', 'routeid', cleaned_sentence)
        cleaned_sentence = re.sub(r'gaotie\w*', 'traintype', cleaned_sentence)
        cleaned_sentence = re.sub(r'dongche\w*', 'traintype', cleaned_sentence)
        cleaned_sentence = re.sub(r'(kuaisu)', 'traintype', cleaned_sentence)
        cleaned_sentence = re.sub(r'(tekuai)', 'traintype', cleaned_sentence)
        cleaned_sentence = re.sub(r'(shang hai|shanghai)', 'cityname', cleaned_sentence)
        cleaned_sentence = re.sub(r'(nan jing|nanjing)', 'cityname', cleaned_sentence)
        cleaned_sentence = re.sub(r'(su zhou|suzhou)', 'cityname', cleaned_sentence)
        cleaned_sentence = re.sub(r'(tai yuan|taiyuan)', 'cityname', cleaned_sentence)
        cleaned_sentence = re.sub(r'(zhen jiang|zhenjiang)', 'cityname', cleaned_sentence)
        cleaned_sentence = re.sub(r'(wu xi|wuxi)', 'cityname', cleaned_sentence)
        cleaned_sentence = re.sub(r'(spicy hot noodles)', 'food', cleaned_sentence)
        cleaned_sentence = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_sentence)
        cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence)
        return cleaned_sentence

    # mp.set_start_method('spawn', force=True)
    def create_bert_sentence_emb(self, sentences, type):
        cache_size = 400
        if type == 'TTspanurl':
            max_length = 17
        if type == 'TTlog':
            max_length = 16

        if type == 'BTspanurl':
            max_length = 14
        if type == 'BTlog':
            max_length = 10


        # Set cache batch size depending on GPU memory
        embeddings = []

        self.Bert_model.eval()

        for i in range(0, len(sentences), cache_size):
            #print(f"Processing bert setence batch starting at index {i}", flush=True)
            batch_sentences = sentences[i: i + cache_size]
            # Tokenize sentences in current batch
            tokenized_batches = self.tokenizer(batch_sentences, truncation=True, padding='max_length',
                                          add_special_tokens=True,
                                          return_tensors='pt', max_length=max_length)

            # Move tokenized input of current batch to GPU
            tokens_tensor = tokenized_batches['input_ids'].to(self.device, non_blocking=True)
            attention_mask = tokenized_batches['attention_mask'].to(self.device, non_blocking=True)

            with torch.no_grad():
                #print(f"Getting outputs for batch starting at index {i}", flush=True)
                outputs = self.Bert_model(tokens_tensor, attention_mask=attention_mask)

            hidden_states = outputs[2]
            token_vecs = hidden_states[-1]
            sentence_embs = torch.mean(token_vecs, dim=1)

            embeddings.append(sentence_embs)
            torch.cuda.empty_cache()
        embeddings = torch.cat(embeddings, dim=0)

        return embeddings.cpu()

