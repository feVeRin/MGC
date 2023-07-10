import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
import PIL.Image as pilimg
import torchvision
import re
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# def get_class(room_code):
#     match = re.match(r"([A-Z]+)([0-9]+)", room_code, re.I)
#     if match:
#         items = match.groups()
#         return items[0]
#     else:
#         return room_code

class Dataset(Data.Dataset):
    def __init__(self):
        # # Ballroom
        # dic = {'Chacha': 0, 'Jive': 1, 'Quickstep': 2, 'Rumba': 3, 'Samba': 4, 'Tango': 5, 'VienneseWaltz': 6, 'Waltz': 7}
        # self.stft = glob.glob('../data/Ballroom/Ballroom_stft/*.npy')
        # self.mel = glob.glob('../data/Ballroom/Ballroom_mel/*.npy')
        # self.mfcc = glob.glob('../data/Ballroom/Ballroom_mfcc/*.npy')
        # self.labels = glob.glob('../data/Ballroom/Ballroom_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/Ballroom/Ballroom_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/Ballroom/Ballroom_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/Ballroom/Ballroom_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/Ballroom/Ballroom_stft/*.npy')

        # # Ballroom-Extended    
        # dic = {'Chacha': 0, 'Foxtrot': 1, 'Jive': 2, 'Pasodoble': 3, 'Quickstep': 4, 'Rumba': 5, 'Salsa': 6, 'Samba': 7, "Slowwaltz": 8, "Tango": 9, "Viennesewaltz": 10, 'Waltz': 11, 'Wcswing': 12}        
        # self.stft = glob.glob('../data/Ballroom-Extended/Ballroom-Extended_stft/*.npy')
        # self.mel = glob.glob('../data/Ballroom-Extended/Ballroom-Extended_mel/*.npy')
        # self.mfcc = glob.glob('../data/Ballroom-Extended/Ballroom-Extended_mfcc/*.npy')
        # self.labels = glob.glob('../data/Ballroom-Extended/Ballroom-Extended_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/Ballroom-Extended/Ballroom-Extended_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/Ballroom-Extended/Ballroom-Extended_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/Ballroom-Extended/Ballroom-Extended_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/Ballroom-Extended/Ballroom-Extended_stft/*.npy')

        ### emoMusic-G
        # dic = {'Blues': 0, 'Classical': 1, 'Country': 2, 'Electronic': 3, 'Folk': 4, 'Jazz': 5, 'Pop': 6, 'Rock': 7}
        # self.stft = glob.glob('../data/emoMusic-G/emoMusic-G_stft/*.npy')
        # self.mel = glob.glob('../data/emoMusic-G/emoMusic-G_mel/*.npy')
        # self.mfcc = glob.glob('../data/emoMusic-G/emoMusic-G_mfcc/*.npy')
        # self.labels = glob.glob('../data/emoMusic-G/emoMusic-G_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/emoMusic-G/emoMusic-G_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/emoMusic-G/emoMusic-G_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/emoMusic-G/emoMusic-G_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/emoMusic-G/emoMusic-G_stft/*.npy')
        
        ## Emotify-G
        # dic = {'0': 0, '1': 1, '2': 2, '3': 3}
        # self.stft = glob.glob('../data/Emotify-G/Emotify-G_stft/*.npy')
        # self.mel = glob.glob('../data/Emotify-G/Emotify-G_mel/*.npy')
        # self.mfcc = glob.glob('../data/Emotify-G/Emotify-G_mfcc/*.npy')
        # self.labels = glob.glob('../data/Emotify-G/Emotify-G_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/Emotify-G/Emotify-G_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/Emotify-G/Emotify-G_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/Emotify-G/Emotify-G_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/Emotify-G/Emotify-G_stft/*.npy')

        # # # FMA-SMALL
        # dic = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hiphop': 3, 'Instrumental': 4, 'International': 5, 'Pop': 6, 'Rock': 7}
        # self.stft = glob.glob('../data/FMA-SMALL/FMA-SMALL_stft/*.npy')
        # self.mel = glob.glob('../data/FMA-SMALL/FMA-SMALL_mel/*.npy')
        # self.mfcc = glob.glob('../data/FMA-SMALL/FMA-SMALL_mfcc/*.npy')
        # self.labels = glob.glob('../data/FMA-SMALL/FMA-SMALL_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/FMA-SMALL/FMA-SMALL_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/FMA-SMALL/FMA-SMALL_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/FMA-SMALL/FMA-SMALL_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/FMA-SMALL/FMA-SMALL_stft/*.npy')

        # # ### GiantStepsKey-G
        # dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22}
        # self.stft = glob.glob('../data/GiantStepsKey-G/GiantStepsKey-G_stft/*.npy')
        # self.mel = glob.glob('../data/GiantStepsKey-G/GiantStepsKey-G_mel/*.npy')
        # self.mfcc = glob.glob('../data/GiantStepsKey-G/GiantStepsKey-G_mfcc/*.npy')
        # self.labels = glob.glob('../data/GiantStepsKey-G/GiantStepsKey-G_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/GiantStepsKey-G/GiantStepsKey-G_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/GiantStepsKey-G/GiantStepsKey-G_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/GiantStepsKey-G/GiantStepsKey-G_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/GiantStepsKey-G/GiantStepsKey-G_stft/*.npy')

        # ### GMD-G
        # dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17}
        # self.stft = glob.glob('../data/GMD-G/GMD-G_stft/*.npy')
        # self.mel = glob.glob('../data/GMD-G/GMD-G_mel/*.npy')
        # self.mfcc = glob.glob('../data/GMD-G/GMD-G_mfcc/*.npy')
        # self.labels = glob.glob('../data/GMD-G/GMD-G_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/GMD-G/GMD-G_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/GMD-G/GMD-G_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/GMD-G/GMD-G_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/GMD-G/GMD-G_stft/*.npy')

        # GTZAN
        dic = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, "reggae": 8, "rock": 9}
        # self.stft = glob.glob('../data/GTZAN/GTZAN_stft/*.npy')
        # self.mel = glob.glob('../data/GTZAN/GTZAN_mel/*.npy')
        # self.mfcc = glob.glob('../data/GTZAN/GTZAN_mfcc/*.npy')
        # self.labels = glob.glob('../data/GTZAN/GTZAN_stft/*.npy')
        self.stft = glob.glob('../data/resizedata/GTZAN/GTZAN_stft/*.npy')
        self.mel = glob.glob('../data/resizedata/GTZAN/GTZAN_mel/*.npy')
        self.mfcc = glob.glob('../data/resizedata/GTZAN/GTZAN_mfcc/*.npy')
        self.labels = glob.glob('../data/resizedata/GTZAN/GTZAN_stft/*.npy')
        
        # # HOMBURG
        # dic = {'alternative': 0, 'blues': 1, 'electronic': 2, 'jazz': 3, 'folkcountry': 4, 'pop': 5, 'funksoulrnb': 6, 'raphiphop': 7, "rock": 8}
        # self.stft = glob.glob('../data/HOMBURG/HOMBURG_stft/*.npy')
        # self.mel = glob.glob('../data/HOMBURG/HOMBURG_mel/*.npy')        
        # self.mfcc = glob.glob('../data/HOMBURG/HOMBURG_mfcc/*.npy')
        # self.labels = glob.glob('../data/HOMBURG/HOMBURG_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/HOMBURG/HOMBURG_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/HOMBURG/HOMBURG_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/HOMBURG/HOMBURG_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/HOMBURG/HOMBURG_stft/*.npy')

        # # ISMIR04
        # dic = {'classical': 0, 'electronic': 1, 'jazz_blues': 2, 'jazz': 2, 'metal': 3, 'metal_punk': 3, 'punk': 3, 'rock': 4, 'pop': 4, 'rock_pop': 4, 'world': 5}
        # self.stft = glob.glob('../data/ISMIR04/ISMIR04_stft/*.npy')
        # self.mel = glob.glob('../data/ISMIR04/ISMIR04_mel/*.npy')
        # self.mfcc = glob.glob('../data/ISMIR04/ISMIR04_mfcc/*.npy')
        # self.labels = glob.glob('../data/ISMIR04/ISMIR04_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/ISMIR04/ISMIR04_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/ISMIR04/ISMIR04_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/ISMIR04/ISMIR04_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/ISMIR04/ISMIR04_stft/*.npy')

        # # MICM
        # dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
        # self.stft = glob.glob('../data/MICM/MICM_stft/*.npy')
        # self.mel = glob.glob('../data/MICM/MICM_mel/*.npy')
        # self.mfcc = glob.glob('../data/MICM/MICM_mfcc/*.npy')
        # self.labels = glob.glob('../data/MICM/MICM_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/MICM/MICM_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/MICM/MICM_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/MICM/MICM_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/MICM/MICM_stft/*.npy')

        # # # Seyerlehner-Unique
        # #soul_rnb -> soul
        # #volksmusik -> folk
        # #work -> spoken
        # dic = {'blues': 0, 'country': 1, 'dance': 2, 'electronica': 3, 'hip-hop': 4, 'jazz': 5, 'klassik': 6, 'reggae': 7, "rock": 8, "schlager": 9, 'soul': 10, 'volksmusik': 11, 'world': 12, 'wort': 13}
        # self.stft = glob.glob('../data/Seyerlehner-Unique/Seyerlehner-Unique_stft/*.npy')
        # self.mel = glob.glob('../data/Seyerlehner-Unique/Seyerlehner-Unique_mel/*.npy')
        # self.mfcc = glob.glob('../data/Seyerlehner-Unique/Seyerlehner-Unique_mfcc/*.npy')
        # self.labels = glob.glob('../data/Seyerlehner-Unique/Seyerlehner-Unique_stft/*.npy')
        # self.stft = glob.glob('./data/resizedata/Seyerlehner-Unique/Seyerlehner-Unique_stft/*.npy')
        # self.mel = glob.glob('./data/resizedata/Seyerlehner-Unique/Seyerlehner-Unique_mel/*.npy')
        # self.mfcc = glob.glob('./data/resizedata/Seyerlehner-Unique/Seyerlehner-Unique_mfcc/*.npy')
        # self.labels = glob.glob('./data/resizedata/Seyerlehner-Unique/Seyerlehner-Unique_stft/*.npy')
        
        for i in range(len(self.labels)):
            self.labels[i] = self.labels[i].split('/')[-1].split('_')[0]
            print(self.labels[i])
            self.labels[i] = dic[self.labels[i]]

    def __getitem__(self, idx):
        self.len = len(self.stft)
        
        return torch.from_numpy(np.load(self.stft[idx])), torch.from_numpy(np.load(self.mel[idx])), torch.from_numpy(np.load(self.mfcc[idx])), self.labels[idx]
        # return torch.from_numpy(np.load(self.stft[idx])), torch.from_numpy(np.load(self.mel[idx])), self.labels[idx]
    
    def __len__(self):
        return len(self.stft)