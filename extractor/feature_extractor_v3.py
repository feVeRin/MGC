from operator import concat
import os
from turtle import st
# from librosa.core import audio
import numpy as np
import librosa
import librosa.display
# import matplotlib.pyplot as plt 
import glob
from matplotlib.pyplot import axis
import csv

#####stft extraction#####
def stft_extraction(path):
    y = librosa.load(path, sr=None)[0]
    stft_result = librosa.stft(y, n_fft=4096, win_length = 4096, hop_length=512)
    D = np.abs(stft_result)
    S_dB = librosa.power_to_db(D, ref=np.max)
    return S_dB

#####mel-spectrogram extraction#####
min_level_db= -100
def normalize_mel(S):
        return np.clip((S-min_level_db)/-min_level_db,0,1)
def mel_extraction(path):
        y = librosa.load(path, sr=None)[0]
        S =  librosa.feature.melspectrogram(y=y, sr=44100)
        norm_log_S = normalize_mel(librosa.power_to_db(S, ref=np.max))
        return norm_log_S

#####mfcc extraction#####
def mfcc_extraction(path):
    y = librosa.load(path, sr=None)[0]
    # mfcc_result = librosa.mfcc(y, n_fft=4096, win_length = 4096, hop_length=512)
    mfcc_result = librosa.feature.mfcc(y, n_fft=4096, win_length = 4096, hop_length=512)
    D = np.abs(mfcc_result)
    S_dB = librosa.power_to_db(D, ref=np.max)
    return S_dB

#####chroma_stft extraction#####
# # using an energy (magnitude) spectrum
def chromastft_extraction(path):
    y,sr = librosa.load(path)
    S = np.abs(librosa.stft(y, n_fft=4096, win_length = 4096, hop_length=512)) # apply short-time fourier transform
    chroma_e = librosa.feature.chroma_stft(S=S, sr=sr)
    return chroma_e

# librosa.feature.spectral_contrast
def contrast_extraction(path):
    y, sr = librosa.load(path)
    S = np.abs(librosa.stft(y, n_fft=4096, win_length = 4096, hop_length=512))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    return contrast

# librosa.feature.spectral_rolloff
def rolloff85_extraction(path):
    y, sr = librosa.load(path)
    # Approximate maximum frequencies with roll_percent=0.85 (default)
    rolloff85 = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return rolloff85
def rolloff95_extraction(path):
    y,sr = librosa.load(path)
    # Approximate maximum frequencies with roll_percent=0.95
    rolloff95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    return rolloff95

# chroma_cqt, tonntz

#mfcc delta, delta-delta
def delta_extraction(path):
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)
    return mfcc_delta
def delta2_extraction(path):
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    return mfcc_delta2

#cqt
def cqt_extraction(path):
    y, sr = librosa.load(path)
    C = np.abs(librosa.cqt(y, sr=sr))
    return C

#     return img
def resize(img_folder, size):    
    img = np.load(img_folder[k])
    img = np.resize(img, (size, 1292))
    img = np.pad(img, ((0,0), (0, 4)))
    img = img /np.linalg.norm(img)

    return img
dataset = ['Ballroom', 'Ballroom-Extended', 'emoMusic-G', 'Emotify-G', 'FMA-SMALL', 'GiantStepsKey-G', 'GMD-G', 'GTZAN', 'HOMBURG', 'ISMIR04', 'MICM', 'Seyerlehner-Unique']
# dataset = ['Ballroom-Extended', 'emoMusic-G', 'Emotify-G', 'FMA-SMALL', 'GiantStepsKey-G', 'GMD-G', 'GTZAN', 'HOMBURG', 'ISMIR04', 'MICM', 'Seyerlehner-Unique']

for num in range(len(dataset)):
    if dataset[num] == 'Ballroom':
        genre = []
        f = open("./Ballroom.csv", "r")
        reader = csv.reader(f)
        for row in reader:
            onehot = row[1]+row[2]+row[3]+row[4]+row[5]+row[6]+row[7]+row[8]
            # print(row)
            if onehot == '10000000':
                genre.append('Chacha')
            elif onehot == '01000000':
                genre.append('Jive')
            elif onehot == '00100000':
                genre.append('Quickstep')
            elif onehot == '00010000':
                genre.append('Rumba')
            elif onehot == '00001000':
                genre.append('Samba')
            elif onehot == '00000100':
                genre.append('Tango')
            elif onehot == '00000010':
                genre.append('VienneseWaltz')
            elif onehot == '00000001':
                genre.append('Waltz')
    elif dataset[num] == 'Ballroom-Extended':
        genre = ['Chacha', 'Foxtrot', 'Jive', 'Pasodoble', 'Quickstep', 'Rumba', 'Salsa', 'Samba', 'Slowwaltz', 'Tango', 'Viennesewaltz', 'Waltz', 'Wcswing']
    elif dataset[num] == 'emoMusic-G':
        genre = ['Blues', 'Classical', 'Country', 'Electronic', 'Folk', 'Jazz', 'Pop', 'Rock']
    elif dataset[num] == 'Emotify-G':
        genre = ['Rock', 'Classical', 'Pop', 'Electronic']
    elif dataset[num] == 'FMA-SMALL':
        genre = []
        f = open("./fma_small.csv", "r")
        reader = csv.reader(f)
        index = 0
        for row in reader:
            onehot = row[1]+row[2]+row[3]+row[4]+row[5]+row[6]+row[7]+row[8]
            # print(row)
            if onehot == '10000000':
                genre.append('Electronic')
            elif onehot == '01000000':
                genre.append('Experimental')
            elif onehot == '00100000':
                genre.append('Folk') 
            elif onehot == '00010000':
                genre.append('Hiphop')
            elif onehot == '00001000':
                genre.append('Instrumental')
            elif onehot == '00000100':
                genre.append('International')
            elif onehot == '00000010':
                genre.append('Pop')
            elif onehot == '00000001':
                genre.append('Rock')
    elif dataset[num] == 'GiantStepsKey-G':
        genre = ['Reggae-dub', 'Chill-out', 'Indie-dance-nu-dsc.', 'Hiphop', 'Glitch-hop', 'Deep-house', 'House', 
            'Tech-house', 'Techno', 'Minimal', 'Funk-r-and-b', 'Pop-rock', 'Drum-and-bass', 'Hardcore-hard-tech.Electronica',
            'Dj-tools', 'Electro-house', 'Hard-dance', 'Dubstep', 'Breaks', 'Trance', 'Psy-trance', 'Progressive-house']
    elif dataset[num] == 'GMD-G':
        genre = ['Afrobeat', 'Afrocuban', 'Blues', 'Country', 'Dance', 'Funk', 'Gospel', 'Highlelife', 'Hiphop',
            'Jazz', 'Latin', 'Middleeastern', 'Neworleans', 'Pop', 'Punk', 'Reggae', 'Rock', 'Soul']
    elif dataset[num] == 'GTZAN':
        genre = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    elif dataset[num] == 'HOMBURG':
        genre = ['Alternative', 'Blues', 'Electronic', 'Jazz', 'Folk/Country', 'Pop', 'Funk/Soul/RnB', 'Rap/Hiphop', 'Rock']
    elif dataset[num] == 'ISMIR04':
        genre = ['Classical', 'Electronic', 'Jazz/Blues', 'Metal/Punk', 'Rock/Pop', 'World']
    elif dataset[num] == 'MICM':
        genre = ['0', '1', '2', '3',  '4', '5', '6']
    elif dataset[num] == 'Seyerlehner-Unique':
        genre = ['Blues', 'Classical', 'Country', 'Dance', 'Electronica', 'Hiphop', 'Jazz', 'Reggae', 'Rock', 'Schlager', 'Soul/RnB', 'Folk', 'World', 'Spoken']

    save_path = './data/' + dataset[num] + '/' + dataset[num]
    audio_path = 'D:/개인연구/music/music_data/' + dataset[num] + '/wav'
    index = 0

    stft_root = './features/stft_ori'+'/' + dataset[num] + '_stft_ori'
    mel_root = './features/mel_ori'+'/' + dataset[num] + '_mel_ori'
    mfcc_root = './features/mfcc_ori'+'/' + dataset[num] + '_mfcc_ori'
    chromastft_root = './features/chromastft_ori'+'/' + dataset[num] + '_chromastft_ori'
    spectralcontrast_root = './features/spectralcontrast_ori'+'/' + dataset[num] + '_spectralcontrast_ori'
    spectralrolloff85_root = './features/spectralrolloff85_ori'+'/' + dataset[num] + '_spectralrolloff85_ori'
    spectralrolloff95_root = './features/spectralrolloff95_ori'+'/' + dataset[num] + '_spectralrolloff95_ori'
    delta_root = './features/delta_ori'+'/' + dataset[num] + '_delta_ori'
    delta2_root = './features/delta2_ori'+'/' + dataset[num] + '_delta2_ori'
    cqt_root = './features/cqt_ori'+'/' + dataset[num] + '_cqt_ori'

    # print(save_path)
    # stop
    if not os.path.isdir('./data/'+dataset[num]):
        os.mkdir('./data/'+dataset[num])
    if not os.path.isdir(stft_root):
        os.makedirs(stft_root)
    if not os.path.isdir(mel_root):
        os.makedirs(mel_root)
    if not os.path.isdir(mfcc_root):
        os.makedirs(mfcc_root)
    if not os.path.isdir(chromastft_root):
        os.makedirs(chromastft_root)
    if not os.path.isdir(spectralcontrast_root):
        os.makedirs(spectralcontrast_root)
    if not os.path.isdir(spectralrolloff85_root):
        os.makedirs(spectralrolloff85_root)
    if not os.path.isdir(spectralrolloff95_root):
        os.makedirs(spectralrolloff95_root)
    if not os.path.isdir(delta_root):
        os.makedirs(delta_root)
    if not os.path.isdir(delta2_root):
        os.makedirs(delta2_root)
    if not os.path.isdir(cqt_root):
        os.makedirs(cqt_root)

    for root, dirs, files in os.walk(audio_path):
        
        for fname in files:
            full_fname = os.path.join(root, fname)
            # a = stft_extraction(full_fname)
            # b = mel_extraction(full_fname)
            # c = mfcc_extraction(full_fname)
            d = chromastft_extraction(full_fname)
            e = contrast_extraction(full_fname)
            f = rolloff85_extraction(full_fname)
            g = rolloff95_extraction(full_fname)
            h = delta_extraction(full_fname)
            i = delta2_extraction(full_fname)
            j = cqt_extraction(full_fname)
            
            if dataset[num] == 'Ballroom' or dataset[num] == 'FMA-SMALL':
                #Ballroom, FMA-SMALL
                ori_name = genre[index] + '_' + fname.split('.')[0]
                index = index + 1
            elif dataset[num] == 'Ballroom-Extended' or dataset[num] == 'emoMusic-G' or dataset[num] == 'Emotify-G' or dataset[num] == 'GiantStepsKey-G' or dataset[num] == 'GMD-G' or dataset[num] == 'GTZAN' or dataset[num] == 'ISMIR04' or dataset[num] == 'HOMBURG' or dataset[num] == 'Seyerlehner-Unique':
                # Ballroom-Extended, GTZAN, ISMIR04
                ori_name = fname.split('.')[0]
            elif dataset[num] == 'MICM':
                # MICM, Emotify-G
                ori_name = fname.split('.')[0][:-4]
                
            # np.save(os.path.join(stft_root, ori_name), a)
            # np.save(os.path.join(mel_root, ori_name), b)
            # np.save(os.path.join(mfcc_root, ori_name), c)
            np.save(os.path.join(chromastft_root, ori_name), d)
            np.save(os.path.join(spectralcontrast_root, ori_name), e)
            np.save(os.path.join(spectralrolloff85_root, ori_name), f)
            np.save(os.path.join(spectralrolloff95_root, ori_name), g)
            np.save(os.path.join(delta_root, ori_name), h)
            np.save(os.path.join(delta2_root, ori_name), i)
            np.save(os.path.join(cqt_root, ori_name), j)
            print(ori_name + ' ori saved')


    ### Create Train data from music files
    # img1_folder = glob.glob(stft_root + '/*.npy')
    # img2_folder = glob.glob(mel_root + '/*.npy')
    # img3_folder = glob.glob(mfcc_root + '/*.npy')
    img4_folder = glob.glob(chromastft_root + '/*.npy')
    img5_folder = glob.glob(spectralcontrast_root + '/*.npy')
    img6_folder = glob.glob(spectralrolloff85_root + '/*.npy')
    img7_folder = glob.glob(spectralrolloff95_root + '/*.npy')
    img8_folder = glob.glob(delta_root + '/*.npy')
    img9_folder = glob.glob(delta2_root + '/*.npy')
    img10_folder = glob.glob(cqt_root + '/*.npy')
    index = 0
    # if not os.path.isdir(save_path + '_stft'):
    #     os.mkdir(save_path + '_stft')
    # if not os.path.isdir(save_path + '_mel'):
    #     os.mkdir(save_path + '_mel')
    # if not os.path.isdir(save_path + '_mfcc'):
    #     os.mkdir(save_path + '_mfcc')
    if not os.path.isdir(save_path + '_chromastft'):
        os.mkdir(save_path + '_chromastft')
    if not os.path.isdir(save_path + '_contrast'):
        os.mkdir(save_path + '_contrast')
    if not os.path.isdir(save_path + '_rolloff85'):
        os.mkdir(save_path + '_rolloff85')
    if not os.path.isdir(save_path + '_rolloff95'):
        os.mkdir(save_path + '_rolloff95')
    if not os.path.isdir(save_path + '_delta'):
        os.mkdir(save_path + '_delta')
    if not os.path.isdir(save_path + '_delta2'):
        os.mkdir(save_path + '_delta2')
    if not os.path.isdir(save_path + '_cqt'):
        os.mkdir(save_path + '_cqt')
    for k in range(len(img4_folder)):
        # img1 = resize(img1_folder, 256)
        # img2 = resize(img2_folder, 256)
        # img3 = resize(img3_folder, 256)
        img4 = resize(img4_folder, 256)
        img5 = resize(img5_folder, 256)
        img6 = resize(img6_folder, 256)
        img7 = resize(img7_folder, 256)
        img8 = resize(img8_folder, 256)
        img9 = resize(img9_folder, 256)
        img10 = resize(img10_folder, 256)
            
        if dataset[num] == 'Ballroom'or dataset[num] == 'FMA-SMALL':
            # Ballroom, FMA-SMALL
            index = index + 1
        # npy_name = img1_folder[k].split('/')[-1].split('\\')[-1][:-4]
        npy_name = img4_folder[k].split('/')[-1].split('\\')[-1][:-4]

        # np.save(save_path + '_stft/' + npy_name, img1)
        # np.save(save_path + '_mel/' + npy_name, img2)
        # np.save(save_path + '_mfcc/' + npy_name, img3)
        np.save(save_path + '_chromastft/' + npy_name, img4)
        np.save(save_path + '_contrast/' + npy_name, img5)
        np.save(save_path + '_rolloff85/' + npy_name, img6)
        np.save(save_path + '_rolloff95/' + npy_name, img7)
        np.save(save_path + '_delta/' + npy_name, img8)
        np.save(save_path + '_delta2/' + npy_name, img9)
        np.save(save_path + '_cqt/' + npy_name, img10)

        print(npy_name + ' resized saved')