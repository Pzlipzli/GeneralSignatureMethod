import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

from DataProcess import SignatureAugment
from Settings import *
warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, t_min, t_day, start, end, path_m, path_d, high, low, process=4):
        """
        :param t_min: minute data window for one day
        :param t_day: daily data window for one day
        :param start: start date
        :param end: end date
        :param path_m: the path of minute data
        :param path_d: the path of daily data
        :param high: the name of the high frequency
        :param low: the name of the low frequency
        :param process: number of processes (accelerate data loading)
        """
        self.t_min = t_min
        self.t_day = t_day
        self.start = start
        self.end = end
        self.path_m = path_m
        self.path_d = path_d
        self.freq_min = high
        self.freq_day = low
        self.process = process

        self.df_min, self.df_day = self.__load_all_data()

        os.makedirs(path_dict['data'], exist_ok=True)

    def __load_all_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        :return: load minute data (high freq) and daily data (low freq)
        """
        min_start_date = (self.start - pd.Timedelta(days=(self.t_min - 1))).strftime('%Y-%m-%d')
        day_start_date = (self.start - pd.Timedelta(days=(self.t_day - 1))).strftime('%Y-%m-%d')
        min_end_date = (self.end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        day_end_date = (self.end).strftime('%Y-%m-%d')

        min_files = [os.path.join(self.path_m, file) for file in os.listdir(self.path_m) if file.endswith('.csv')
                     and min_start_date <= file[:-4] <= min_end_date]
        day_files = [os.path.join(self.path_d, file) for file in os.listdir(self.path_d) if file.endswith('.csv')
                    and day_start_date <= file[:-4] <= day_end_date]

        df_min = self.__load_paths(min_files, freq=self.freq_min)
        df_day = self.__load_paths(day_files, freq=self.freq_day)

        return df_min, df_day

    def __load_paths(self, path_list: list, freq: str) -> pd.DataFrame:
        """
        :param path_list: a list of paths
        :return: a dataframe of all the data (volume is log transformed)
        """
        dfs_all = []
        path_list.sort()

        for file_path in path_list:
            df = pd.read_csv(file_path)
            df = df[['开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量', '货币对']]  # according to the format of the data
            df.columns = ['openTime', 'open', 'high', 'low', 'close', 'volume', 'token']
            df['openTime'] = pd.to_datetime(df['openTime'])
            df.sort_values(by=['token', 'openTime'], inplace=True)

            start_dt = datetime.strptime(file_path[-14:-4], '%Y-%m-%d') + pd.Timedelta(hours=8)\
                if freq == self.freq_day else datetime.strptime(file_path[-14:-4], '%Y-%m-%d')

            df_empty = self.get_empty_df(start_dt, freq)
            df = df.groupby('token').apply(self.empty_process, empty=df_empty)
            df.reset_index(drop=True, inplace=True)

            df['volume'] = np.log(df['volume'] + 0.1)
            dfs_all.append(df)

        data = pd.concat(dfs_all, axis=0)
        data.sort_values(by=['token', 'openTime'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data

    @staticmethod
    def empty_process(group, empty):
        """
        front-fill empty df
        :param group: groupby token
        :param empty: empty df
        :return: front-filled df
        """
        group.reset_index(drop=True, inplace=True)
        df_merge = pd.merge(empty, group, how='left', left_on='openTime', right_on='openTime')
        return df_merge.ffill()

    @staticmethod
    def get_empty_df(start: datetime, frequency: str):
        """
        create empty df, used for front-fill
        :param start: start date
        :param frequency: frequency
        : return: empty df
        """
        open_list = pd.date_range(start=start, end=start + pd.Timedelta(days=1), freq=frequency, inclusive="left")

        df_empty = pd.DataFrame({"openTime": open_list})
        df_empty["closeTime"] = df_empty["openTime"] + pd.Timedelta(frequency) - pd.Timedelta(seconds=1)

        return df_empty

    def run(self):
        """
        create factors with signature according to date
        save factors to csv files
        """
        dates = pd.date_range(start=self.start, end=self.end).tolist()  # get all dates
        dates = [dt + pd.Timedelta(hours=8) for dt in dates]

        for dt in tqdm(dates):
            data_min = self.df_min[((dt - pd.Timedelta(days=self.t_min - 1)) <= self.df_min['openTime'])
                              & (self.df_min['closeTime'] <= (dt + pd.Timedelta(days=1)))]
            data_day = self.df_day[((dt - pd.Timedelta(days=self.t_day - 1)) <= self.df_day['openTime'])
                              & (self.df_day['closeTime'] <= (dt + pd.Timedelta(days=1)))]

            past_token = data_day[data_day['openTime'] == (dt - pd.Timedelta(days=self.t_day - 1))].token.unique()
            now_token = data_day[data_day['openTime'] == dt].token.unique()
            tokens = list(set(now_token) & set(past_token))

            with ProcessPoolExecutor(max_workers=self.process) as executor:
                sig_with_args = partial(self.signature, data_min=data_min, data_day=data_day)
                futures = [executor.submit(sig_with_args, token) for token
                           in tokens]
                df_sig_all = [future.result() for future in as_completed(futures)]

            df_sig = pd.concat(df_sig_all, ignore_index=True)
            df_sig = df_sig.loc[:, (df_sig != df_sig.iloc[0]).any()]  # delete constant columns
            df_sig.columns = [f'alpha_{i}' for i in range(df_sig.shape[1] - 1)] + ['token']
            df_sig['openTime'] = dt.replace(hour=8, minute=0, second=0)
            df_sig['closeTime'] = (dt + pd.Timedelta(days=1)).replace(hour=7, minute=59, second=59)
            df_sig.sort_values(by=['token', 'openTime'], inplace=True)

            df_sig.to_csv(path_dict['data'] + f'{dt.strftime("%Y-%m-%d")}.csv', index=False)
            print(f'{dt.strftime("%Y-%m-%d")} is saved.')

    def signature(self, token, data_min, data_day):
        """
        calculate signature of each token
        :param token
        :param data_min: high freq data
        :param data_day: low freq data
        :return: signature of the given token
        """
        data_token_min = data_min[data_min['token'] == token].drop(columns=['token', 'openTime']).reset_index(
            drop=True).drop(columns=['closeTime'])
        data_token_day = data_day[data_day['token'] == token].drop(columns=['token', 'openTime']).reset_index(
            drop=True).drop(columns=['closeTime'])


        last_close_min = data_token_min.iloc[-1, 3]
        last_close_day = data_token_day.iloc[-1, 3]

        data_token_min[['open', 'close', 'high', 'low']] = data_token_min[['open', 'close', 'high',
                                                                           'low']] / last_close_min
        data_token_day[['open', 'close', 'high', 'low']] = data_token_day[['open', 'close', 'high',
                                                                           'low']] / last_close_day

        sig_aug = SignatureAugment()
        concatenated_sig1 = sig_aug.run_augment(data_token_min, sig_depth=4, projection='pair', slides=(360, 60), window='slide')
        concatenated_sig2 = sig_aug.run_augment(data_token_day, sig_depth=4, projection='pair', window='all', scale='post')

        final_concatenated_sig = np.concatenate([concatenated_sig1, concatenated_sig2])

        df_final_sig = pd.DataFrame(final_concatenated_sig).T
        df_final_sig['token'] = token

        return df_final_sig


def generate_date_ranges(start_date=None, end_date=None, mode='generate'):
    """
    make sure that only 2 months of data are generated
    """
    file_list = sorted(os.listdir(path_dict['data']))
    if mode == 'generate':
        assert start_date and end_date, 'start_date and end_date must be provided in generate mode'
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    elif mode == 'update' and file_list:  # if the mode is update and there are files in the directory
        start_date = pd.to_datetime(file_list[-1][:-4]) + pd.Timedelta(days=1)
        end_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d 00:00:00')) - pd.Timedelta(days=1)
    else:
        raise ValueError("'update' is not for first time running, try 'generate' instead")

    current_start = start_date
    while current_start <= end_date:
        current_end = current_start + pd.DateOffset(months=2) - pd.Timedelta(days=1)

        if current_end > end_date:
            current_end = end_date

        yield current_start, current_end

        current_start = current_end + pd.Timedelta(days=1)


if __name__ == '__main__':
    high_freq = '5min'
    low_freq = '1D'
    path_m = path_dict[f'{high_freq}']
    path_d = path_dict[f'{low_freq}']
    t_m = 20  # the range of high freq data
    t_d = 60  # the range of low freq data
    start = '2019-07-01'
    end = '2019-07-20'
    process = 4  # max number of processes
    mode = 'update'  # update or generate

    for start, end in tqdm(generate_date_ranges(start, end, mode)):
        dataloader = DataLoader(t_m, t_d, start, end, path_m, path_d, high_freq, low_freq, process)
        dataloader.run()
        del dataloader
