import argparse
from datetime import datetime
import os
import shutil
from typing import List, Any

import pyworld

from utility import *

FEATURE_DIM = 36  # 特征维度
SAMPLE_RATE = 16000  # 音频采样率
FRAMES = 512  # 帧数
FFTSIZE = 1024  # 快速傅里叶变换值
SPEAKERS_NUM = len(speakers) # 语音条数
CHUNK_SIZE = 5  # concate CHUNK_SIZE audio clips together
EPSILON = 1e-10
MODEL_NAME = 'starganvc_model'


def load_wavs(dataset: str, sr) :
    '''
    data dict contains all audios file path &
    resdict contains all wav files   
    '''
    data = {}
    with os.scandir(dataset) as it :
        for entry in it :
            if entry.is_dir() :
                data[entry.name] = []
                # print(entry.name, entry.path)
                with os.scandir(entry.path) as it_f :
                    for onefile in it_f :
                        if onefile.is_file() :
                            # print(onefile.path)
                            data[entry.name].append(onefile.path)
    print(f'loaded keys: {data.keys()}')
    # data like {TM1:[xx,xx,xxx,xxx]}
    resdict = {}

    cnt = 0
    for key, value in data.items() :
        resdict[key] = {}
        # 此处的one_file是一条完整的路径
        for one_file in value :
            filename = one_file.split('/')[-1].split('.')[0]  # like 100061
            newkey = f'{filename}'
            # sr = sample rate，采样率
            wav, _ = librosa.load(one_file, sr=sr, mono=True, dtype=np.float64)

            resdict[key][newkey] = wav
            # resdict[key].append(temp_dict) #like TM1:{100062:[xxxxx], .... }
            print('.', end='')
            cnt += 1

    print(f'\nTotal {cnt} audio files!')
    return resdict

# 对一整段音频矩阵进行切片处理
def chunks(iterable, size) :
    """Yield successive n-sized chunks from iterable."""
    # 根据size切片
    for i in range(0, len(iterable), size) :
        yield iterable[i : i + size]


# 提取 梅尔倒谱系数
def wav_to_mcep_file(dataset: str, sr=SAMPLE_RATE, processed_filepath: str = './data/processed') :
    '''convert wavs to mcep feature using image repr'''
    shutil.rmtree(processed_filepath)  # 递归地清除目录树
    os.makedirs(processed_filepath, exist_ok=True)
    # glob返回目录下所有符合条件的文件
    allwavs_cnt = len(glob.glob(f'{dataset}/*/*.wav'))
    print(f'Total {allwavs_cnt} audio files!')

    d = load_wavs(dataset, sr)
    for one_speaker in d.keys() :
        values_of_one_speaker = list(d[one_speaker].values())
        # 调用上文的chunks函数，里面使用了yield的生成器，所以要套enumerate
        for index, one_chunk in enumerate(chunks(values_of_one_speaker, CHUNK_SIZE)) :
            wav_concated = []  # preserve one batch of wavs
            temp = one_chunk.copy() # 使用copy是为了防止覆盖操作

            # concate wavs
            for one in temp :
                wav_concated.extend(one)
            # 由数组转化为numpy数组
            wav_concated = np.array(wav_concated)

            # process one batch of wavs
            # f0：基频
            # ap：非周期信号参数
            # sp：频谱包络
            # cal_mcep 的定义在下方
            f0, ap, sp, coded_sp = cal_mcep(wav_concated, sr=sr, dim=FEATURE_DIM)
            newname = f'{one_speaker}_{index}'
            file_path_z = os.path.join(processed_filepath, newname)
            # save several arrays into one file ended with npz
            np.savez(file_path_z, f0=f0, coded_sp=coded_sp)
            print(f'[save]: {file_path_z}')

            # split mcep t0 multi files
            for start_idx in range(0, coded_sp.shape[1] - FRAMES + 1, FRAMES) :
                one_audio_seg = coded_sp[:, start_idx : start_idx + FRAMES]

                if one_audio_seg.shape[1] == FRAMES :
                    temp_name = f'{newname}_{start_idx}'
                    filePath = os.path.join(processed_filepath, temp_name)

                    np.save(filePath, one_audio_seg)
                    print(f'[save]: {filePath}.npy')


def cal_mcep(wav, sr=SAMPLE_RATE, dim=FEATURE_DIM, fft_size=FFTSIZE) :
    ''' 
        cal mcep given wav singnal
        the frame_period used only for pad_wav_to_get_fixed_frames
    '''
    f0, timeaxis, sp, ap, coded_sp = world_features(wav, sr, fft_size, dim)
    coded_sp = coded_sp.T  # dim x n

    return f0, ap, sp, coded_sp


def world_features(wav, sr, fft_size, dim) :
    # 将采集的基频限制在了[71, 800]Hz
    # timeaxis 是每一帧的时间轴坐标
    f0, timeaxis = pyworld.harvest(wav, sr, f0_floor=71.0, f0_ceil=800.0)
    # cheaptrick算法 计算频谱包络
    sp = pyworld.cheaptrick(wav, f0, timeaxis, sr, fft_size=fft_size)
    # 计算非周期
    ap = pyworld.d4c(wav, f0, timeaxis, sr, fft_size=fft_size)
    # 频谱包络
    coded_sp = pyworld.code_spectral_envelope(sp, sr, dim)

    return f0, timeaxis, sp, ap, coded_sp



if __name__ == "__main__" :
    start = datetime.now()
    parser = argparse.ArgumentParser(description='Convert the wav waveform to mel-cepstral coefficients(MCCs)\
    and calculate the speech statistical characteristics')

    input_dir = './data/speakers'
    output_dir = './data/processed'

    parser.add_argument('--input_dir', type=str, help='the direcotry contains data need to be processed',
                        default=input_dir)
    parser.add_argument('--output_dir', type=str, help='the directory stores the processed data', default=output_dir)

    argv = parser.parse_args()
    input_dir = argv.input_dir
    output_dir = argv.output_dir

    os.makedirs(output_dir, exist_ok=True)

    wav_to_mcep_file(input_dir, SAMPLE_RATE, processed_filepath=output_dir)

    # input_dir is train dataset. we need to calculate and save the speech \
    # statistical characteristics for each speaker.
    # 这个类从utility引用而来，给每个说话人定义数据上的特征
    generator = GenerateStatistics(output_dir)
    generator.generate_stats()
    generator.normalize_dataset()
    end = datetime.now()
    print(f"[Runing Time]: {end - start}")
