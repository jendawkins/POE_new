import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class dataReader:
    
    def __init__(self, fname = '20180210-  Updated Bacteria and phage raw concentrations for travis (1).xlsk'):
        self.fname = fname
        self.data = pd.read_excel('mouse_data.xlsx',header = [0,1])
        # self.header_upper = pd.read_excel()
        self.bacteria, self.phages = self.make_dict()
        self.times = self.data['Unnamed: 1_level_0']['Time (d)']


    def make_dict(self):
        keep_bacteria = self.data.columns.levels[0][self.data.columns.levels[0].str.contains('(relative abundance)')]
        keep_phages = self.data.columns.levels[0][self.data.columns.levels[0].str.contains('qPCR')][1:]
        mlabs = ['M1','M2','M3','M4','M5']
        self.all_phages = [np.sum(self.data[k][s] for k in keep_phages[1:]) for s in mlabs]
        
        d = dict()
        for  k in keep_bacteria:
            d[str.replace(k,' (relative abundance)','')] = self.data[k].iloc[:,:5]
        
        p = dict()
        for k in keep_phages[1:]:
            # import pdb; pdb.set_trace()
            p[str.split(k)[2]] = np.divide(self.data[k].iloc[:,:5],self.data[keep_phages[0]].iloc[:,:5])
        return d,p
    
    def plot_changes(self,m = None,b = None,p = None):
        if m == 'mean':
            plt.figure()
            for k,v in self.bacteria.items():
                m = np.nanmean(v,1)
                s = np.nanstd(v,1)
                plt.errorbar(self.times,m,yerr = s,label = k)        
                plt.legend()
                plt.title('Bacteria')
            plt.show()
            plt.figure()
            for k,v in self.phages.items():
                m = np.nanmean(v,1)
                s = np.nanstd(v,1)
                plt.errorbar(self.times,m,yerr = s,label = k)        
                plt.legend()
                plt.title('Phages')
            plt.show()
        if m == None:
            m = list(self.bacteria.values())[0].columns  
        if b == None:
            b = self.bacteria.keys()
        if p == None:
            p = self.phages.keys()
