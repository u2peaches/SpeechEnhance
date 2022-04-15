import torch
import torch.nn as nn
import numpy as np
from model import Generator
from hparams import hparams
from dataset import emphasis
import glob
import soundfile as sf
import os
import librosa
import matplotlib.pyplot as plt
from numpy.linalg import norm


def enh_segan(model, noisy, para):
    win_len = para.win_len

    N_slice = len(noisy) // win_len

    # 对音频不能整切割的部分进行补充
    if not len(noisy) % win_len == 0:
        short = win_len - len(noisy) % win_len
        temp_noisy = np.pad(noisy, (0, short), 'wrap')  # 使用数据的前一小部分去扩充后面不足的
        N_slice = N_slice + 1

    slices = temp_noisy.reshape(N_slice, win_len)
    enh_slice = np.zeros(slices.shape)

    for n in range(N_slice):
        m_slice = slices[n]

        # 预加重
        m_slice = emphasis(m_slice)

        m_slice = np.expand_dims(m_slice, axis=(0, 1))

        m_slice = torch.from_numpy(m_slice)

        z = nn.init.normal_(torch.Tensor(1, para.size_z[0], para.size_z[1]))

        model.eval()
        with torch.no_grad():
            generated_slice = model(m_slice, z)
        generated_slice = generated_slice.numpy()

        # 反预加重
        generated_slice = emphasis(generated_slice[0, 0, :], pre=False)
        enh_slice[n] = generated_slice

    # 将信号转为1维输出
    enh_speech = enh_slice.reshape(N_slice * win_len)
    return enh_speech[:len(noisy)]


""" 求信噪比 """


def get_snr(clean, noisy):
    noise = noisy - clean
    snr = 20 * np.log(norm(clean) / (norm(noisy) + 1e-7))
    return snr


if __name__ == "__main__":
    para = hparams()

    path_eval = 'eval147'
    os.makedirs(path_eval, exist_ok=True)

    # 加载模型
    n_epoch = 0
    model_file = "save/voicebank/2022-4-14_RMSProp/G_21_0.4042.pkl"

    generator = Generator()
    generator.load_state_dict(torch.load(model_file, map_location='cpu'))

    path_test_clean = 'dataset/voicebank/clean_testset_wav/'
    path_test_noisy = 'dataset/voicebank/noisy_testset_wav/'

    """ 对中文语音的训练 """
    # path_test_clean = 'dataset/data_thchs30/train'
    # path_test_noisy = 'dataset/data_thchs30/train_noisy\CAFE-CAFE-2/0/'

    test_clean_wavs = glob.glob(path_test_clean + '/*wav')

    w = 0
    fs = para.fs
    for clean_file in test_clean_wavs:
        name = os.path.split(clean_file)[-1]
        noisy_file = os.path.join(path_test_noisy, name)
        if not os.path.isfile(noisy_file):
            continue
        if w == 10:
            break
        w += 1
        #  读取干净语音
        clean, _ = librosa.load(clean_file, sr=fs, mono=True)
        noisy, _ = librosa.load(noisy_file, sr=fs, mono=True)

        snr = get_snr(clean, noisy)
        print("%s  snr=%.2f" % (noisy_file, snr))

        if snr <= 0:
            # print('processing &s with snr %.2f' % (noisy_file, snr))

            # 获取增强语音
            enh = enh_segan(generator, noisy, para)

            # 语音保存
            sf.write(os.path.join(path_eval, 'noisy-' + name), noisy, fs)
            sf.write(os.path.join(path_eval, 'clean-' + name), clean, fs)
            sf.write(os.path.join(path_eval, 'enh-' + name), enh, fs)

            # 画频谱图
            fig_name = os.path.join(path_eval, name[:-4] + '-' + str(n_epoch) + ".jpg")

            plt.subplot(3, 1, 1)
            plt.specgram(clean, NFFT=512, Fs=fs)
            plt.xlabel("clean specgram")
            plt.subplot(3, 1, 2)
            plt.specgram(noisy, NFFT=512, Fs=fs)
            plt.subplot(3, 1, 3)
            plt.specgram(enh, NFFT=512, Fs=fs)
            plt.xlabel("enhance specgram")
            plt.savefig(fig_name)
