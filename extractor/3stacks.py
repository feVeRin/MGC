import os
from turtle import st
from librosa.core import audio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt 
import gc
import glob
from matplotlib.pyplot import axis
import re
import csv

### Ballroom
# dataset = 'Ballroom'
# genre = ['Chacha', 'Jive', 'Quickstep', 'Rumba', 'Samba', 'Tango', 'Viennesewaltz', 'Waltz']
# genre = []
# f = open("./Ballroom.csv", "r")
# reader = csv.reader(f)
# for row in reader:
#     onehot = row[1]+row[2]+row[3]+row[4]+row[5]+row[6]+row[7]+row[8]
#     # print(row)
#     if onehot == '10000000':
#         genre.append('Chacha')
#     elif onehot == '01000000':
#         genre.append('Jive')
#     elif onehot == '00100000':
#         genre.append('Quickstep')
#     elif onehot == '00010000':
#         genre.append('Rumba')
#     elif onehot == '00001000':
#         genre.append('Samba')
#     elif onehot == '00000100':
#         genre.append('Tango')
#     elif onehot == '00000010':
#         genre.append('VienneseWaltz')
#     elif onehot == '00000001':
#         genre.append('Waltz')

### Ballroom-Extended
# dataset = 'Ballroom-Extended'
# genre = ['Chacha', 'Foxtrot', 'Jive', 'Pasodoble', 'Quickstep', 'Rumba', 'Salsa', 'Samba', 'Slowwaltz', 'Tango', 'Viennesewaltz', 'Waltz', 'Wcswing']

### FMA-MEDIUM 
# dataset = 'FMA-MEDIUM'
# # genre = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Experimental', 'Folk', 'Hiphop',
# #           'Instrumental', 'International', 'Jazz', 'Old-Time&Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
# genre = []
# f = open("./fma_medium.csv", "r")
# reader = csv.reader(f)
# index = 0
# for row in reader:
#     onehot = row[0]+row[1]+row[2]+row[3]+row[4]+row[5]+row[6]+row[7]+row[8]+row[9]+row[10]+row[11]+row[12]+row[13]+row[14]+row[15]
#     if onehot == '1000000000000000':
#         genre.append('Blues')
#     elif onehot == '0100000000000000':
#         genre.append('Classical')
#     elif onehot == '0010000000000000':
#         genre.append('Country')
#     elif onehot == '0001000000000000':
#         genre.append('Easy Listening')
#     elif onehot == '0000100000000000':
#         genre.append('Electronic')
#     elif onehot == '0000010000000000':
#         genre.append('Experimental')
#     elif onehot == '0000001000000000':
#         genre.append('Folk')
#     elif onehot == '0000000100000000':
#         genre.append('Hiphop')
#     elif onehot == '0000000010000000':
#         genre.append('Instrumental')
#     elif onehot == '0000000001000000':
#         genre.append('International')
#     elif onehot == '0000000000100000':
#         genre.append('Jazz')
#     elif onehot == '0000000000010000':
#         genre.append('Old-Time&Historic')
#     elif onehot == '0000000000001000':
#         genre.append('Pop')
#     elif onehot == '0000000000000100':
#         genre.append('Rock')
#     elif onehot == '0000000000000010':
#         genre.append('Soul-RnB')
#     elif onehot == '0000000000000001':
#         genre.append('Spoken')

# ### FMA-SMALL 
# dataset = 'FMA-SMALL'
# # genre = ['Electronic', 'Experimental', 'Folk', 'Hiphop', 'Instrumental', 'International', 'Pop', 'Rock']
# genre = []
# f = open("./fma_small.csv", "r")
# reader = csv.reader(f)
# index = 0
# for row in reader:
#     onehot = row[1]+row[2]+row[3]+row[4]+row[5]+row[6]+row[7]+row[8]
#     # print(row)
#     if onehot == '10000000':
#         genre.append('Electronic')
#     elif onehot == '01000000':
#         genre.append('Experimental')
#     elif onehot == '00100000':
#         genre.append('Folk') 
#     elif onehot == '00010000':
#         genre.append('Hiphop')
#     elif onehot == '00001000':
#         genre.append('Instrumental')
#     elif onehot == '00000100':
#         genre.append('International')
#     elif onehot == '00000010':
#         genre.append('Pop')
#     elif onehot == '00000001':
#         genre.append('Rock')

# ## GTZAN
# dataset = 'GTZAN'
# genre = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# # HOMBURG
# dataset = 'HOMBURG'
# genre = ['Alternative', 'Blues', 'Electronic', 'Jazz', 'Folk/Country', 'Pop', 'Funk/Soul/RnB', 'Rap/Hiphop', 'Rock']

# # # ISMIR04
# dataset = 'ISMIR04'
# genre = ['Classical',
#          'Electronic',
#          'Jazz/Blues',
#          'Metal/Punk',
#          'Rock/Pop',
#          'World']

### MICM
# dataset = 'MICM'
# genre = ['0', '1', '2', '3',  '4', '5', '6']

# Seyerlehner-Unique
# dataset = 'Seyerlehner-Unique'
# genre = ['Blues', 'Classical', 'Country', 'Dance', 'Electronica', 'Hiphop', 'Jazz', 'Reggae', 'Rock', 'Schlager', 'Soul/RnB', 'Folk', 'World', 'Spoken']

### emoMusic-G
dataset = 'emoMusic-G'
genre = ['Blues', 'Classical', 'Country', 'Electronic', 'Folk', 'Jazz', 'Pop', 'Rock']

# ### Emotify-G
# dataset = 'Emotify-G'
# genre = ['0', '1', '2', '3']

# ### GiantStepsKey-G
# dataset = 'GiantStepsKey-G'
# genre = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']

### GMD-G
# dataset = 'GMD-G'
# genre = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']

save_path = 'data/' + dataset + '/' + dataset 
audio_path = 'D:/개인연구/music/music_data/' + dataset + '/wav'
index = 0
# print(genre)
#####stft extraction#####
stft_root = './stft_ori'+'/' + dataset + '_stft_ori'
def stft_extraction(path):
    y = librosa.load(path, sr=None)[0]
    stft_result = librosa.stft(y, n_fft=4096, win_length = 4096, hop_length=512)
    D = np.abs(stft_result)
    S_dB = librosa.power_to_db(D, ref=np.max)
    return S_dB

#####mel-spectrogram extraction#####
mel_root = './mel_ori'+'/' + dataset + '_mel_ori'
min_level_db= -100
def normalize_mel(S):
        return np.clip((S-min_level_db)/-min_level_db,0,1)
def mel_extraction(path):
        y = librosa.load(path, sr=None)[0]
        S =  librosa.feature.melspectrogram(y=y, sr=44100)
        norm_log_S = normalize_mel(librosa.power_to_db(S, ref=np.max))
        # print(np.shape(y))
        # print(np.shape(S))
        # print(np.shape(norm_log_S))
        return norm_log_S

#####mfcc extraction#####
mfcc_root = './mfcc_ori'+'/' + dataset + '_mfcc_ori'
def mfcc_extraction(path):
    y = librosa.load(path, sr=None)[0]
    # mfcc_result = librosa.mfcc(y, n_fft=4096, win_length = 4096, hop_length=512)
    mfcc_result = librosa.feature.mfcc(y, n_fft=4096, win_length = 4096, hop_length=512)
    D = np.abs(mfcc_result)
    S_dB = librosa.power_to_db(D, ref=np.max)
    return S_dB

def resize(img_folder, size):    
    img = np.load(img_folder[k])
    img = np.resize(img, (size, 1292))
    img = np.pad(img, ((0,0), (0, 4)))
    img = img /np.linalg.norm(img)

    return img
def stacking(img):
    section1 = np.expand_dims(img[:, :432], axis=0)
    section2 = np.expand_dims(img[:, 432:864], axis=0)
    section3 = np.expand_dims(img[:, 864:], axis=0)

    layer1 = np.append(section1, section2, axis=0)
    layer2 = np.append(layer1, section3, axis=0)

    return layer2
    
if not os.path.isdir(stft_root):
    os.mkdir(stft_root)
if not os.path.isdir(mel_root):
    os.mkdir(mel_root)
if not os.path.isdir(mfcc_root):
    os.mkdir(mfcc_root)

for root, dirs, files in os.walk(audio_path):
    for fname in files:        
        full_fname = os.path.join(root, fname)
        a = stft_extraction(full_fname)
        b = mel_extraction(full_fname)
        c = mfcc_extraction(full_fname)
        
        if dataset == 'Ballroom' or dataset == 'FMA-MEDIUM' or dataset == 'FMA-SMALL':
            #Ballroom, FMA-SMALL, FMA-MEDIUM
            ori_name = genre[index] + '_' + fname.split('.')[0]
            index = index + 1
        elif dataset == 'Ballroom-Extended' or dataset == 'GTZAN' or dataset == 'ISMIR04' or dataset == 'HOMBURG' or dataset == 'Seyerlehner-Unique' or dataset == 'emoMusic-G' or dataset == 'Emotify-G' or dataset == 'GiantStepsKey-G' or dataset == 'GMD-G':
            # Ballroom-Extended, GTZAN, ISMIR04, HOMBURG, Seyerlehner-Unique, emoMusic-G, Emotify-G, GiantStepsKey-G, GMD-G
            ori_name = fname.split('.')[0]
        elif dataset == 'MICM':
            # MICM
            ori_name = fname.split('.')[0][:-4]
        elif dataset == 'Tropical Genres':
            fname = fname[:-8] + '_' + fname[-8:]
            ori_name = fname.split('.')[0]
            
        np.save(os.path.join(stft_root, ori_name), a)
        np.save(os.path.join(mel_root, ori_name), b)
        np.save(os.path.join(mfcc_root, ori_name), c)
        print(ori_name + ' ori saved')


### Create Train data from music files
img1_folder = glob.glob(stft_root + '/*.npy')
img2_folder = glob.glob(mel_root + '/*.npy')
img3_folder = glob.glob(mfcc_root + '/*.npy')
index = 0

if not os.path.isdir('./data/'+dataset):
    os.mkdir('./data/'+dataset)
if not os.path.isdir(save_path + '_stft'):        
    os.mkdir(save_path + '_stft')
if not os.path.isdir(save_path + '_mel'):        
    os.mkdir(save_path + '_mel')
if not os.path.isdir(save_path + '_mfcc'):        
    os.mkdir(save_path + '_mfcc')
# print(img3_folder)
for k in range(len(img3_folder)):
    img1 = resize(img1_folder, 256)
    img2 = resize(img2_folder, 128)
    img3 = resize(img3_folder, 96)
    
    stacked_stft = stacking(img1)
    stacked_mel = stacking(img2)
    stacked_mfcc = stacking(img3)
    
    if dataset == 'Ballroom' or dataset == 'FMA-MEDIUM' or dataset == 'FMA-SMALL':
        # Ballroom, FMA-MEDIUM, FMA-SMALL
        # ori_name = genre[index] + '_' + fname.split('.')[0]
        index = index + 1
    npy_name = img3_folder[k].split('/')[-1].split('\\')[-1][:-4]
    
    np.save(save_path + '_stft/' + npy_name, stacked_stft)
    np.save(save_path + '_mel/' + npy_name, stacked_mel)
    np.save(save_path + '_mfcc/' + npy_name, stacked_mfcc)

    print(npy_name + ' stacked saved')