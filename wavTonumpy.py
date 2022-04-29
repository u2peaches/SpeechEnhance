import numpy as np
import librosa
import os
from viocebank_data_generation import wav_split, save_slices
""" 完成wav语音的分段 """


if __name__ == "__main__":

    # 未处理数据集存放位置
    clean_wav_path = "dataset/data_thchs30/train/"
    noisy_wav_path = "dataset/data_thchs30/train_noisy/"

    # 处理后数据集存放位置
    catch_train_clean = "dataset/segan/1/clean/"
    catch_train_noisy = "dataset/segan/1/noisy/"

    os.makedirs(catch_train_clean, exist_ok=True)
    os.makedirs(catch_train_noisy, exist_ok=True)

    win_length = 16384  # 语音切段段长
    strid = int(win_length / 2)  # 语音切段步长

    # train_segan_chinese.scp用于存储处理好后的clean语音以及noisy语音对
    with open("dataset/train_segan_chinese.scp", 'wt') as f:
        for root, dirs, files in os.walk(clean_wav_path):
            for file in files:
                file_clean_name = os.path.join(root, file)
                name = os.path.split(file_clean_name)[-1]  # 获取语音文件的名字
                if name.endswith("wav"):
                    print("processing file %s" % (file_clean_name))
                    for root1, dirs1, files1 in os.walk(noisy_wav_path):
                        file_noisy_name = os.path.join(root1, name)
                        file_noisy_name = file_noisy_name.replace("\\", "/")
                        if not os.path.exists(file_noisy_name):
                            continue
                        clean_data, sr = librosa.load(file_clean_name, sr=16000, mono=True)
                        noisy_data, sr = librosa.load(file_noisy_name, sr=16000, mono=True)
                        if not len(clean_data) == len(noisy_data):
                            print("file length are not equal")
                            continue

                        # 语音分段+保存
                        clean_slices = wav_split(clean_data, win_length, strid)
                        clean_namelist = save_slices(clean_slices, os.path.join(catch_train_clean, name))

                        snr = os.path.split(root1)[-1]  # 获取语音文件的名字

                        # 创建噪声语音分段存储文件夹
                        catch_train_path, _ = os.path.split(os.path.join(catch_train_noisy, snr, name))
                        os.makedirs(catch_train_path, exist_ok=True)

                        # 语音分段+保存
                        noisy_slices = wav_split(noisy_data, win_length, strid)
                        noisy_namelist = save_slices(noisy_slices, os.path.join(catch_train_noisy, snr + "/" + name))
                        for clean_catch_name, noisy_catch_name in zip(clean_namelist, noisy_namelist):
                            f.write("%s %s\n" % (clean_catch_name, noisy_catch_name))
                        continue
