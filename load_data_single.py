import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
import PIL.Image as pilimg
import torchvision
import re
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def get_class(room_code):
    match = re.match(r"([A-Z]+)([0-9]+)", room_code, re.I)
    if match:
        items = match.groups()
        return items[0]
    else:
        return room_code

class Dataset(Data.Dataset):
    def __init__(self):
        # # Ballroom
        # dic = {'Chacha': 0, 'Jive': 1, 'Quickstep': 2, 'Rumba': 3, 'Samba': 4, 'Tango': 5, 'VienneseWaltz': 6, 'Waltz': 7}
        # self.mel = glob.glob('./data/Ballroom/Ballroom_mel/*.npy')
        # self.labels = glob.glob('./data/Ballroom/Ballroom_mel/*.npy') 

        # # # Ballroom-Extended    
        # dic = {'Chacha': 0, 'Foxtrot': 1, 'Jive': 2, 'Pasodoble': 3, 'Quickstep': 4, 'Rumba': 5, 'Salsa': 6, 'Samba': 7, "Slowwaltz": 8, "Tango": 9, "Viennesewaltz": 10, 'Waltz': 11, 'Wcswing': 12}        
        # self.mel = glob.glob('./data/Ballroom-Extended/Ballroom-Extended_mel/*.npy')
        # self.labels = glob.glob('./data/Ballroom-Extended/Ballroom-Extended_mel/*.npy')

        ### emoMusic-G
        # dic = {'Blues': 0, 'Classical': 1, 'Country': 2, 'Electronic': 3, 'Folk': 4, 'Jazz': 5, 'Pop': 6, 'Rock': 7}
        # self.mel = glob.glob('./data/emoMusic-G/emoMusic-G_mel/*.npy')
        # self.labels = glob.glob('./data/emoMusic-G/emoMusic-G_mel/*.npy')

        # ### Emotify-G
        # dic = {'0': 0, '1': 1, '2': 2, '3': 3}
        # self.mel = glob.glob('./data/Emotify-G/Emotify-G_mel/*.npy')
        # self.labels = glob.glob('./data/Emotify-G/Emotify-G_mel/*.npy')

        # # FMA-MEDIUM
        # dic = {'Blues': 0, 'Classical': 1, 'Country': 2, 'Easy Listening': 3, 'Electronic': 4, 'Experimental': 5, 'Folk': 6, 'Hiphop': 7, 'Instrumental': 8, 'International': 9, 'Jazz': 10, 'Old-Time&Historic': 11, 'Pop': 12, 'Rock': 13, 'Soul-RnB': 14, 'Spoken': 15}        
        # self.mel = glob.glob('./data/FMA-MEDIUM/FMA-MEDIUM_mel/*.npy')
        # self.labels = glob.glob('./data/FMA-MEDIUM/FMA-MEDIUM_mel/*.npy')

        # # # FMA-SMALL
        # dic = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hiphop': 3, 'Instrumental': 4, 'International': 5, 'Pop': 6, 'Rock': 7}        
        # self.mel = glob.glob('./data/FMA-SMALL/FMA-SMALL_mel/*.npy')
        # self.labels = glob.glob('./data/FMA-SMALL/FMA-SMALL_mel/*.npy')

        # # ### GiantStepsKey-G
        # dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22}
        # self.mel = glob.glob('./data/GiantStepsKey-G/GiantStepsKey-G_mel/*.npy')
        # self.labels = glob.glob('./data/GiantStepsKey-G/GiantStepsKey-G_mel/*.npy')

        # ### GMD-G
        # dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17}
        # self.mel = glob.glob('./data/GMD-G/GMD-G_mel/*.npy')
        # self.labels = glob.glob('./data/GMD-G/GMD-G_mel/*.npy')

        # # GTZAN       
        dic = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, "reggae": 8, "rock": 9}
        self.mel = glob.glob('./data/GTZAN/GTZAN_mel/*.npy')
        self.labels = glob.glob('./data/GTZAN/GTZAN_mel/*.npy')

        # # # HOMBURG
        # dic = {'alternative': 0, 'blues': 1, 'electronic': 2, 'jazz': 3, 'folkcountry': 4, 'pop': 5, 'funksoulrnb': 6, 'raphiphop': 7, "rock": 8}
        # self.mel = glob.glob('./data/HOMBURG/HOMBURG_mel/*.npy')
        # self.labels = glob.glob('./data/HOMBURG/HOMBURG_mel/*.npy')

        # # # ISMIR04
        # dic = {'classical': 0, 'electronic': 1, 'jazz_blues': 2, 'jazz': 2, 'metal': 3, 'metal_punk': 3, 'punk': 3, 'rock': 4, 'pop': 4, 'rock_pop': 4, 'world': 5}
        # self.mel = glob.glob('./data/ISMIR04/ISMIR04_mel/*.npy')
        # self.labels = glob.glob('./data/ISMIR04/ISMIR04_mel/*.npy')

        # # MICM
        # dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
        # self.mel = glob.glob('./data/MICM/MICM_mel/*.npy')
        # self.labels = glob.glob('./data/MICM/MICM_mel/*.npy')

        # # Seyerlehner-Unique
        # dic = {'blues': 0, 'country': 1, 'dance': 2, 'electronica': 3, 'hip-hop': 4, 'jazz': 5, 'klassik': 6, 'reggae': 7, "rock": 8, "schlager": 9, 'soul': 10, 'volksmusik': 11, 'world': 12, 'wort': 13}
        # #soul_rnb -> soul
        # #volksmusik -> folk
        # #work -> spoken
        # self.mel = glob.glob('./data/Seyerlehner-Unique/Seyerlehner-Unique_mel/*.npy')
        # self.labels = glob.glob('./data/Seyerlehner-Unique/Seyerlehner-Unique_mel/*.npy')

        # Tropical Genres
        # dic = {'bachata': 0, 'cumbia': 1, 'merengue': 2, 'salsa': 3, 'vallenato': 4}
        # self.mel = glob.glob('./data/Tropical Genres/Tropical Genres_mel/*.npy')
        # self.labels = glob.glob('./data/Tropical Genres/Tropical Genres_mel/*.npy')

        for i in range(len(self.labels)):
            self.labels[i] = self.labels[i].split('\\')[-1].split('_')[0]
            self.labels[i] = dic[self.labels[i]]

        
    ###ResNet50_trust
    # def __getitem__(self, idx):
    #     self.len = len(self.mel)
    #     label_range = np.arange(9).reshape(9,-1)
    #     label_list = np.array(self.labels).reshape(-1,1)
    #     enc = OneHotEncoder()
        
    #     label_onehot = enc.fit_transform(np.array(self.labels).reshape(-1,1))
    #     label_onehot = torch.tensor(label_onehot.toarray())
        
    #     label_unidis = torch.distributions.uniform.Uniform(0, 10).sample(self.labels)
    #     return torch.from_numpy(np.load(self.mel[idx])), self.labels[idx], self.labels[idx], self.labels[idx]

    def __getitem__(self, idx):
        self.len = len(self.mel)    
        
        return torch.from_numpy(np.load(self.mel[idx])), self.labels[idx]

    def __len__(self):
        return len(self.mel)

    # def __getitem__(self, idx):
    #     self.len = len(self.mfcc)    
        
    #     return torch.from_numpy(np.load(self.mfcc[idx])), self.labels[idx]

    # def __len__(self):
    #     return len(self.mfcc)

