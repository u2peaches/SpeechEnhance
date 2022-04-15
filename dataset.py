import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from hparams import hparams
import librosa
import random
import soundfile as sf


class SEGAN_Dataset(Dataset):

    def __init__(self, para):
        self.file_scp = para.train_scp
        files = np.loadtxt(self.file_scp, dtype='str')  # 读取data_generation中生成的clean语音以及noisy语音对
        self.clean_files = files[:, 0].tolist()
        self.noisy_files = files[:, 1].tolist()

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # 读取clean语音数据并进行预加重
        clean_wav = np.load(self.clean_files[idx])
        clean_wav = emphasis(clean_wav)

        # 读取noisy语音数据并进行预加重
        noisy_wav = np.load(self.noisy_files[idx])
        noisy_wav = emphasis(noisy_wav)

        clean_wav = torch.from_numpy(clean_wav)
        noisy_wav = torch.from_numpy(noisy_wav)

        # 增加一个维度
        clean_wav = clean_wav.reshape(1, -1)
        noisy_wav = noisy_wav.reshape(1, -1)

        return clean_wav, noisy_wav

    """  """

    def ref_batch(self, batch_size):
        index = np.random.choice(len(self.clean_files), batch_size).tolist()

        catch_clean = [emphasis(np.load(self.clean_files[i])) for i in index]
        catch_noisy = [emphasis(np.load(self.noisy_files[i])) for i in index]

        catch_clean = np.expand_dims(np.array(catch_clean), axis=1)
        catch_noisy = np.expand_dims(np.array(catch_noisy), axis=1)

        batch_wav = np.concatenate((catch_clean, catch_noisy), axis=1)
        return torch.from_numpy(batch_wav)


""" 对语音数据进行预加重 """


def emphasis(signal, emph_coeff=0.95, pre=True):
    if pre:
        result = np.append(signal[0], signal[1:] - emph_coeff * signal[:-1])
    else:
        result = np.append(signal[0], signal[1:] + emph_coeff * signal[:-1])

    return result
