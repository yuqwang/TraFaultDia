from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class DatasetFusion(Dataset):
    def __init__(self, datamodel, num_traces, span_max_length, log_max_length):
        """
                     Args:
                        span_file_path (string): Path to the csv file1.
                        log_file_path (string): Path to the csv file2.
                        """
        self.num_traces = num_traces
        self.span_max_length = span_max_length
        self.log_max_length = log_max_length

        span_dir_path = './Data/span_formatted'
        log_dir_path = './Data/log_clean/'

        if datamodel == 'train':
            selected_spans_files = [f for f in os.listdir(span_dir_path) if
                                    f.startswith('SpanF') and f.endswith('.csv') and 30 < int(f[5:7]) < 34]
            #Trainticket: F01-F30 are fault categories. > 30 are unlabled traces
            #OnlineBoutique: F01-F32 are fault categories. > 32 are unlabled traces
            selected_logs_files = [f for f in os.listdir(log_dir_path) if
                                   f.startswith('F') and f.endswith('.csv') and 30 < int(f[1:3]) < 34]

            print("datamodel is train:", selected_spans_files)

        if datamodel == 'val':
            selected_spans_files = [f for f in os.listdir(span_dir_path)
                                    if f.startswith('SpanF')
                                    and f.endswith('.csv')
                                    and  int(f[5:7]) > 36]
            # Trainticket: F01-F30 are fault categories. > 36 selects unlabled traces from datasets and not overlap with train sets
            # OnlineBoutique: F01-F32 are fault categories. > 32 selects unlabled traces from datasets and not overlap with train sets
            selected_logs_files = [f for f in os.listdir(log_dir_path)
                                   if f.startswith('F')
                                   and f.endswith('.csv')
                                   and int(f[1:3]) > 36]
            print("datamodel is val:", selected_spans_files)

        self.per_num_trace = int(self.num_traces / len(selected_spans_files))
        self.selected_traces = np.array([])


        self.spans_sample_df = pd.DataFrame()
        self.logs_df = pd.DataFrame()

        for filename in selected_spans_files:
            filepath = os.path.join(span_dir_path, filename)
            df = pd.read_csv(filepath)
            unique_trace_Id = df['TraceId'].unique()
            single_selected_traces = np.random.choice(unique_trace_Id, size=self.per_num_trace, replace=False)
            self.selected_traces = np.concatenate((self.selected_traces, single_selected_traces))
            selected_sample_df = df[df['TraceId'].isin(single_selected_traces)]
            self.spans_sample_df = self.spans_sample_df._append(selected_sample_df, ignore_index=True)


        for filename in selected_logs_files:
            filepath = os.path.join(log_dir_path, filename)
            df = pd.read_csv(filepath)
            self.logs_df = self.logs_df._append(df, ignore_index=True)

        self.spans_sample_df = self.spans_sample_df.copy()
        # unique_traces1 = self.spans_sample_df['TraceId'].unique()

        selected_span_features = ['URL', 'TraceId', 'SpanId', 'Normalized_StartTime', 'Normalized_EndTime',
                                  'Normalized_tree_span_ids']
        #self.spans_sample_df['URL'] = self.spans_sample_df['URL'].apply(lambda x: self.clean_spanurl_tt(str(x)))
        self.spans_sample_df = self.spans_sample_df[selected_span_features]
        self.spans_sample_df = self.spans_sample_df.groupby('TraceId', group_keys=False).nth(
            list(range(self.span_max_length)))
        self.spans_sample_df = self.spans_sample_df.reset_index()
        self.spans_sampleFull_array = self.spans_sample_df.values

        self.logs_sample_df = self.logs_df[self.logs_df['TraceId'].isin(self.selected_traces)]
        self.logs_sample_df = self.logs_sample_df.copy()
        self.logs_sample_df['SpanIdFull'] = self.logs_sample_df['SpanIdFull'].str.replace(']]', '', regex=False)
        # unique_traces = self.logs_sample_df['TraceId'].unique()
        selected_log_features = ['TraceId', 'SpanIdFull', 'LogMsgFull']
        self.logs_sample_df = self.logs_sample_df[selected_log_features]
        self.logs_sample_df = self.logs_sample_df.groupby('TraceId', group_keys=False).nth(
            list(range(self.log_max_length)))
        self.logs_sample_df = self.logs_sample_df.reset_index()
        self.log_sample_array = self.logs_sample_df.values

    def __getitem__(self, index):
        trace_id = self.selected_traces[index]
        spans_sample = self.spans_sampleFull_array[self.spans_sampleFull_array[:, 2] == trace_id]
        logs_sample = self.log_sample_array[self.log_sample_array[:, 1] == trace_id]
        return spans_sample, logs_sample

    def __len__(self):
        return len(self.selected_traces)

    def get_full_data(self):
        return self.spans_sample_df, self.logs_sample_df




