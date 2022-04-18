import os
import random
import librosa
from viocebank_data_generation import wav_split, save_slices
from scipy.io import wavfile
import numpy as np
import soundfile as sf


""" 将噪音分段，前50%用于合成训练噪音，后50%用于合成测试噪音"""

def section(noise_path, noises):
    for noise in noises:
        sample_rate, sig = wavfile.read(os.path.join(noise_path, noise+'.wav'))
        half = sig.shape[0] / 2
        data_train = sig[:int(half)]
        data_test = sig[int(half):]

        sf.write(os.path.join(noise_path, noise + '_train.wav'), data_train, sample_rate)
        sf.write(os.path.join(noise_path, noise + '_test.wav'), data_test, sample_rate)


""" 按照一定的信噪比进行纯净语音和噪音的合成 """

def signal_by_db(speech, noise, snr):
    # # speech = speech.astype(np.int16)
    # # noise = noise.astype(np.int16)
    # print("speech:%s\n" % speech)
    # print("noise:%s\n" % noise)
    len_speech = speech.shape[0]
    len_noise = noise.shape[0]
    #  随机选取噪声音频中音频段
    start = random.randint(0, len_noise - len_speech)
    end = start + len_speech

    add_noise = noise[start:end]

    # 平方求和
    sum_s = np.sum(speech ** 2)
    sum_n = np.sum(add_noise ** 2)
    # 信噪比为-5dB时的权重
    x = np.sqrt(sum_s / (sum_n * pow(10, snr/10)))

    noise = x * add_noise
    target = speech + noise

    return target


if __name__ == "__main__":
    noise_path = 'dataset/qut-noise/'
    noises = ['CAFE-CAFE-1', 'CAFE-CAFE-2', 'CAFE-FOODCOURTB-1', 'CAFE-FOODCOURTB-2', 'CAR-WINDOWNB-1',
              'CAR-WINDOWNB-2', 'CAR-WINUPB-1', 'CAR-WINUPB-2', 'HOME-KITCHEN-1', 'HOME-KITCHEN-2',
              'HOME-LIVINGB-1', 'HOME-LIVINGB-2', 'STREET-CITY-1', 'STREET-CITY-2', 'STREET-KG-1', 'STREET-KG-2']
    # section(noise_path, noises)  # 对噪音数据进行对半切分处理

    # 未处理数据集存放位置
    clean_wav_path = "dataset/data_thchs30/train/"
    noisy_wav_path = "dataset/data_thchs30/train_noisy/"

    # 处理后数据集存放位置
    catch_train_clean = "dataset/segan/data_thchs30/clean/"
    catch_train_noisy = "dataset/segan/data_thchs30/noisy/"

    snrs = [2.5, 7.5, 12.5, 17.5]

    win_length = 16384  # 语音切段段长
    strid = int(win_length / 2)  # 语音切段步长

    change_next = 0  # 用于控制
    num_person = 0  # 读取的说话人数

    # train_segan_chinese.scp用于存储处理好后的clean语音以及noisy语音对
    with open("dataset/train_segan_chinese.scp", 'wt') as f:
        for root, dirs, files in os.walk(clean_wav_path):

            current_person = ''  # 当前说话人

            for file in files:

                #  下面一段用于控制最多只提取当前说话人50句话
                if change_next == 0:
                    current_person = file.split("_")[0]
                if file.split("_")[0] != current_person:
                    num_person += 1
                    change_next = 0
                    print("num_person : %s" % num_person)
                    continue
                if change_next == 40:
                    continue
                change_next += 1

                file_clean_name = os.path.join(root, file)  # 纯净语音位置
                name = os.path.split(file_clean_name)[-1]  # 获取语音文件的名字
                if name.endswith("wav"):
                    index_noisy = random.randint(0, len(noises)-1)  # 随机加入一种噪音
                    noise_file = os.path.join(noise_path, noises[index_noisy] + '_train.wav')  # 最后面加的是测试噪声还是训练噪声

                    noise_data, sr = librosa.load(noise_file, sr=16000, mono=True)
                    clean_data, sr = librosa.load(file_clean_name, sr=16000, mono=True)

                    print("processing file %s" % file_clean_name)
                    for snr in snrs:
                        noisy_file = os.path.join(noisy_wav_path, noises[index_noisy], str(snr), name)  # noisyfile的存储位置
                        noisy_path, _ = os.path.split(noisy_file)
                        os.makedirs(noisy_path, exist_ok=True)
                        mix = signal_by_db(clean_data, noise_data, snr)  # 按照信噪比将噪音同纯净语音混合
                        # noisy_data = np.asarray(mix, dtype=np.int16)
                        noisy_data = mix
                        sf.write(noisy_file, noisy_data, sr)
                        # f.write('%s %s\n' % (noisy_file, file_clean_name))



                        if not len(clean_data) == len(noisy_data):
                            print("file length are not equal")
                            continue

                        # 语音分段+保存
                        clean_slices = wav_split(clean_data, win_length, strid)
                        clean_namelist = save_slices(clean_slices, os.path.join(catch_train_clean, name))

                        # 创建噪声语音分段存储文件夹
                        catch_train_path, _ = os.path.split(os.path.join(catch_train_noisy, noises[index_noisy], str(snr), name))
                        os.makedirs(catch_train_path, exist_ok=True)

                        # 语音分段+保存
                        noisy_slices = wav_split(mix, win_length, strid)
                        noisy_namelist = save_slices(noisy_slices, os.path.join(catch_train_noisy, noises[index_noisy]+"/"+str(snr)+"/"+name))

                        for clean_catch_name, noisy_catch_name in zip(clean_namelist, noisy_namelist):
                            f.write("%s %s\n" % (clean_catch_name, noisy_catch_name))
