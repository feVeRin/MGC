import os
from librosa.core import audio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt 
import gc
# import cv2
import glob
from matplotlib.pyplot import axis
import re
import csv

### Ballroom
dataset = 'Ballroom'
genre = ['Chacha', 'Jive', 'Quickstep', 'Rumba', 'Samba', 'Tango', 'Viennesewaltz', 'Waltz']
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

### Ballroom-Extended
# dataset = 'Ballroom-Extended'
# genre = ['Chacha', 'Foxtrot', 'Jive', 'Pasodoble', 'Quickstep', 'Rumba', 'Salsa', 'Samba', 'Slowwaltz', 'Tango', 'Viennesewaltz', 'Waltz', 'Wcswing']

# emoMusic-G
# dataset = 'emoMusic-G' 
# genre = ['Blues', 'Classical', 'Country', 'Folk', 'Electronic', 'Jazz', 'Pop', 'Rock']

# Emotify-G 
dataset = 'Emotify-G'
genre = ['Rock', 'Classical', 'Pop', 'Electronic']

# ### FMA-MEDIUM 
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

### FMA-SMALL 
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

# GiantStepsKey-G
dataset = 'GiantStepsKey-G'
genre = ['Reggae-dub', 'Chill-out', 'Indie-dance-nu-dsc.', 'Hiphop', 'Glitch-hop', 'Deep-house', 'House', 
          'Tech-house', 'Techno', 'Minimal', 'Funk-r-and-b', 'Pop-rock', 'Drum-and-bass', 'Hardcore-hard-tech.Electronica',
          'Dj-tools', 'Electro-house', 'Hard-dance', 'Dubstep', 'Breaks', 'Trance', 'Psy-trance', 'Progressive-house']

# GMD-G
# dataset = 'GMD-G'
# genre = ['Afrobeat', 'Afrocuban', 'Blues', 'Country', 'Dance', 'Funk', 'Gospel', 'Highlife', 'Hiphop',
#           'Jazz', 'Latin', 'Middleeastern', 'Neworleans', 'Pop', 'Punk', 'Reggae', 'Rock', 'Soul']

# ## GTZAN
# dataset = 'GTZAN'
# genre = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# HOMBURG
# dataset = 'HOMBURG'
# genre = ['Alternative', 'Blues', 'Electronic', 'Jazz', 'Folk/Country', 'Pop', 'Funk/Soul/RnB', 'Rap/Hiphop', 'Rock']

# ISMIR04
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

### Tropical Genres
# dataset = 'Tropical Genres'
# genre = ['bachata',
#         'cumbia',
#         'merengue',
#         'salsa',
#         'vallenato'
#         ]


save_path = 'data/' + dataset + '/' + dataset 
audio_path = 'C:/Users/CAU_MI/Desktop/개인연구/music/music_data/' + dataset + '/wav'
ori_root = './mel_ori'+'/' + dataset + '_mel_ori'
index = 0
# print(genre)
# def feature_extraction(path):
#     y = librosa.load(path, sr=None)[0]
#     mel_result = librosa.mel(y, n_fft=4096, win_length = 4096, hop_length=512)
#     D = np.abs(mel_result)
#     S_dB = librosa.power_to_db(D, ref=np.max)
#     return S_dB
min_level_db= -100
def normalize_mel(S):
        return np.clip((S-min_level_db)/-min_level_db,0,1)
def feature_extraction(path):
        y = librosa.load(path, sr=None)[0]
        S =  librosa.feature.melspectrogram(y=y, sr=44100)
        norm_log_S = normalize_mel(librosa.power_to_db(S, ref=np.max))
        # print(np.shape(y))
        # print(np.shape(S))
        # print(np.shape(norm_log_S))
        return norm_log_S

if not os.path.isdir(ori_root):
            os.mkdir(ori_root)

for root, dirs, files in os.walk(audio_path):
    for fname in files:        
        full_fname = os.path.join(root, fname)
        a = feature_extraction(full_fname)
        
        if dataset == 'Ballroom' or dataset == 'FMA-MEDIUM' or dataset == 'FMA-SMALL':
            #Ballroom, FMA-SMALL, FMA-MEDIUM
            ori_name = genre[index] + '_' + fname.split('.')[0]
            index = index + 1
        elif dataset == 'Ballroom-Extended' or dataset == 'GTZAN' or dataset == 'ISMIR04' or dataset == 'HOMBURG' or dataset == 'Seyerlehner-Unique':
            # Ballroom-Extended, GTZAN, ISMIR04
            ori_name = fname.split('.')[0]
        elif dataset == 'MICM':
            # MICM
            ori_name = fname.split('.')[0][:-4]
        elif dataset == 'Tropical Genres':
            fname = fname[:-8] + '_' + fname[-8:]
            ori_name = fname.split('.')[0]
        # print(os.path.join(ori_root, ori_name))
        np.save(os.path.join(ori_root, ori_name), a)
        print(ori_name + ' ori saved')


### Create mel Train data from music files
img1_folder = glob.glob(ori_root + '/*.npy')
index = 0

for k in range(len(img1_folder)):
    img = np.load(img1_folder[k])
    img = np.resize(img, (128, 1292))
    img = np.pad(img, ((0,0), (0, 4)))
    img = img /np.linalg.norm(img)
    
    section1 = np.expand_dims(img[:, :432], axis=0)
    section2 = np.expand_dims(img[:, 216:648], axis=0)
    section3 = np.expand_dims(img[:, 432:864], axis=0)
    section4 = np.expand_dims(img[:, 648:1080], axis=0)
    section5 = np.expand_dims(img[:, 864:], axis=0)

    layer1 = np.append(section1, section2, axis=0)
    layer2 = np.append(layer1, section3, axis=0)
    layer3 = np.append(layer2, section4, axis=0)
    layer4 = np.append(layer3, section5, axis=0)

    if not os.path.isdir(save_path + '_mel'):
        os.mkdir('./data/'+dataset)
        os.mkdir(save_path + '_mel')
    if dataset == 'Ballroom' or dataset == 'FMA-MEDIUM' or dataset == 'FMA-SMALL':
        # Ballroom, FMA-MEDIUM, FMA-SMALL
        # ori_name = genre[index] + '_' + fname.split('.')[0]
        index = index + 1
    # Ballroom-Extended, GTZAN, HOMBURG, ISMIR04, MICM, Seyerlehner-Unique, Tropical Genres
    
    npy_name = img1_folder[k].split('/')[-1].split('\\')[-1][:-4]
    # print(save_path + '_mel/' + npy_name)
    np.save(save_pat6h + '_mel/' + npy_name, layer4)
    print(npy_name + ' stacked saved')