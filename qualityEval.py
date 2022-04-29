#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2020/9/13 16:53
@Author  : kingback
@Site    :
@File    : SEPM.py
@Software: PyCharm
'''

import pysepm
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import glob
from hparams import hparams
'''
the install order is :

    0. All the package I use the commond "python setup.py install" to install to our machine.
    Maybe we also need to install other packages like :   cython ,numba   ,scipy  ,matplotlib ,numpy  ...

    1. install the gammatone from : https://github.com/detly/gammatone

    2.install the SRMRpy from : https://github.com/jfsantos/SRMRpy

    3.install the python-pesq from  :   https://github.com/vBaiCai/python-pesq

    4.install the pystoi from   :   https://github.com/mpariente/pystoi

    5.install the pyseqm from   :   https://github.com/schmiph2/pysepm

    6. if  you have finished the last step  ,don't forget to change the next step's setup.py .
    This is done so that you do not have to repeatedly download the installed files.
    And there's also the possibility of an error.

    7.  if you run your code facing the ploblem :   
        "
        File "/home/king/anaconda3/envs/tf_gpu/lib/python3.6/site-packages/pysepm-0.1-py3.6.egg/pysepm/__init__.py", line 4, in <module>
        File "/home/king/anaconda3/envs/tf_gpu/lib/python3.6/site-packages/pysepm-0.1-py3.6.egg/pysepm/qualityMeasures.py", line 4, in <module>
        ModuleNotFoundError: No module named 'pesq'
        "

        you can go to your file path "/home/king/anaconda3/envs/tf_gpu/lib/python3.6/site-packages/pysepm-0.1-py3.6.egg/pysepm/qualityMeasures.py"

        find the commond    :
            #import pesq as pypesq # https://github.com/ludlows/python-pesq
        change it like blow :
            import pypesq

    8.  if you run the method pesq & Composite , it may be come some errors because we haven't introduced PESQ packages properly yet.

        you can go to your file path "/home/king/anaconda3/envs/tf_gpu/lib/python3.6/site-packages/pysepm-0.1-py3.6.egg/pysepm/qualityMeasures.py"

        (1).add the commond :
            "from pypesq import pesq as pq"

        (2).replace the following functions from line 347 to line 359:
            "
                def pesq(clean_speech, processed_speech, fs):
                    if fs == 8000:
                        mos_lqo = pq(clean_speech,processed_speech, fs)
                        pesq_mos = 46607/14945 - (2000*np.log(1/(mos_lqo/4 - 999/4000) - 1))/2989#0.999 + ( 4.999-0.999 ) / ( 1+np.exp(-1.4945*pesq_mos+4.6607) )
                    elif fs == 16000:
                         mos_lqo = pq(clean_speech,processed_speech, fs)
                        pesq_mos = np.NaN
                    else:
                        raise ValueError('fs must be either 8 kHz or 16 kHz')
                    return pesq_mos,mos_lqo
            "
        (3).save & quit

'''
para = hparams()
clean_wavs = 'qualityEval/clean_wav/'
denoised_wavs = 'qualityEval/denoised_wav/'


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.wav':
                L.append(os.path.join(root, file))
    return L


# # get wav_lists
# clean_lists = file_name(clean_wavs)
# denoised_lists = file_name(denoised_wavs)
# # Package files
# zipped = zip(clean_lists, denoised_lists)

SegSNRs = []
LLRs = []
WSSs = []
STOIs = []
PESQs = []
CDs = []
scores = []
CSIGs = []
CBAKs = []
COVLs = []

test_clean_wavs = glob.glob(clean_wavs + '/*wav')

fs = para.fs
for clean_file in test_clean_wavs:
    name = os.path.split(clean_file)[-1]
    clean_file = clean_file.replace("\\", "/")
    print("process on %s" % (clean_file))
    for root, dirs, files in os.walk(denoised_wavs):
        file_noisy_name = os.path.join(root, 'enh-'+name[6:])
        file_noisy_name = file_noisy_name.replace("\\", "/")

        if not os.path.exists(file_noisy_name):
            continue

        # Gain speech parameters
        # print(clean_wav, denoised_wav)
        ref, sr0 = sf.read(clean_file, 16000)
        deg, sr1 = sf.read(file_noisy_name, 16000)

        '''
        # Method 1: SNRseg (分段信噪比)
            # from pysepm Call SNRseg to calculate its metrics
            # in this case we can choose our frame length =0.03*1000=30 ms , and the overlap =30 ms *0.75 =22.5 ms
            # The higher the score, the better the performance.
        '''
        SegSNR = pysepm.SNRseg(ref, deg, sr0)
        SegSNRs.append(SegSNR)

        '''
        # Method 2: llr (对数似然比测度)
            # 
            # The higher the score, the better the performance.
        '''
        LLR = pysepm.llr(ref, deg, sr0)
        LLRs.append(LLR)

        '''
        # Method 3: WSS (加权谱倾斜测度)
            # 
            # The smaller the score, the better the performance.
        '''
        WSS = pysepm.wss(ref, deg, sr0)
        WSSs.append(WSS)

        '''
        # Method 4: STOI (可短时客观可懂)
            # 
            # the score from 0-1 . The higher the score, the better the performance.
        '''
        STOI = pysepm.stoi(ref, deg, sr0)
        STOIs.append(STOI)

        '''
        # Method 5: PESQ
            # when I try this commond , I faced some troubles   , so finally I gave up this commond,
            # use the PESQ.py to instead
            # The score from -0.5 - 4.5 .The higher the score, the better the performance. 
        '''

        #  有些音频格式不对
        try:
            NaN, PESQ = pysepm.pesq(ref, deg, sr0)
            PESQs.append(PESQ)
        except:
            continue
        '''
        # Method 6: CD (Cepstrum Distance)
            # 
            # The higher the score, the better the performance.
        '''
        CD = pysepm.cepstrum_distance(ref, deg, sr0)
        CDs.append(CD)

        '''
           # Method 7: LSD (对数谱距离)
                # This method I use the LSD.py to calculate the distance 
                # The smaller the score, the better the performance.      
        '''
        # LSD = pysepm.l

        '''
        Method 1 - 7 use this score to print
        '''
        # score append to scores
        # scores.append(score)

        '''
        # Method 8: Composite
            # In this method , It comes some errors, if you want to solve the error ,  see the step 8 in this file.
            # CSIG , CBAK , COVL all from 1 - 5 , The higher the score, the better the performance.
        '''
        CSIG, CBAK, COVL = pysepm.composite(ref, deg, sr0)
        CSIGs.append(CSIG)
        CBAKs.append(CBAK)
        COVLs.append(COVL)
# print(scores)
print('The average SegSNR evaluation is : %.4f' % (sum(SegSNRs) / len(SegSNRs)))
print('The average LLR evaluation is : %.4f' % (sum(LLRs) / len(LLRs)))
print('The average WSS evaluation is : %.4f' % (sum(WSSs) / len(WSSs)))
print('The average STOI evaluation is : %.4f' % (sum(STOIs) / len(STOIs)))
print('The average PESQ evaluation is : %.4f' % (sum(PESQs) / len(PESQs)))
print('The average CD evaluation is : %.4f' % (sum(CDs) / len(CDs)))
# calculate the standard deviation & variance of the scores
# print('The standard deviation is : %.4f' % (np.std(scores)))
# print('The variance is : %.4f' % (np.var(scores)))

# print(CSIGs, CBAKs, COVLs)
print('The average CSIG evaluation is : %.4f' % (sum(CSIGs) / len(CSIGs)))
print('The average CBAK evaluation is : %.4f' % (sum(CBAKs) / len(CBAKs)))
print('The average COVL evaluation is : %.4f' % (sum(COVLs) / len(COVLs)))

'''
Reference:  
    1.  https://github.com/schmiph2/pysepm/blob/master/examples/examplesForCalculatingMeasures.ipynb    (How to use the packages of SEPM)
    2.  https://zhuanlan.zhihu.com/p/190146707  (音频质量客观评价指标)

'''
