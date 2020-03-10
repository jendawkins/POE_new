import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class dataReader:
    
    def __init__(self, fname = 'mouse_data.xlsx'):
        self.fname = fname
        self.data = pd.read_excel('mouse_data.xlsx',header = [0,1],nrows = 40)
        self.mouse_names = list(set([m for b,m in list(self.data.columns)]))
        self.mouse_names.sort()
        self.mouse_names = self.mouse_names[0:5]
        self.bacteria, self.phages = self.make_dict()
        self.times = self.data['Unnamed: 1_level_0']['Time (d)']
        self.pairs = [('E coli','T4'),('C sporogenes','F1'),('E faecalis','VD13'),('B fragilis','B40-8')]


    def make_dict(self):
        keep_bacteria = self.data.columns.levels[0][self.data.columns.levels[0].str.contains('(cfu/g stool)')]
        keep_bacteria = list(set(keep_bacteria) - set('qPCR of 16s rRNA'))
        keep_phages = self.data.columns.levels[0][self.data.columns.levels[0].str.contains('qPCR')][1:]

        d = dict()
        for  k in keep_bacteria[1:]:
            d[str.replace(k,' (cfu/g stool)','')] = self.data[k].iloc[:,:5]
        p = dict()
        for k in keep_phages:
            p[str.split(k)[2]] = self.data[k].iloc[:,:5]
        return d,p
    
    def plot_changes(self,plot_basic = True, clean = False,plot_pairs = True):
        b = self.bacteria
        p = self.phages
        mice = self.mouse_names
        
        if plot_basic:
            it = 0
            jet= plt.get_cmap('jet')
            colors = iter(jet(np.linspace(0,1,10)))
            plt.figure()
            for k,v in b.items():
                m = np.nanmean(v,1)
                s = np.nanstd(v,1)
                c = next(colors)
                plt.errorbar(self.times,m,yerr = s,color = c, label = k)
                if not clean:
                    m = b[k]
                    plt.plot(self.times,m,color = c, linewidth = .5)    
                plt.yscale('log')
                plt.legend()
                plt.title('Bacteria Abundances over Time')
                plt.ylabel('cfu/g stool')
                plt.xlabel('Time (d)')
                it += 1
            plt.show()
            plt.figure()
            colors = iter(jet(np.linspace(0,1,10)))
            for k,v in p.items():
                m = np.nanmean(v,1)
                s = np.nanstd(v,1)
                c = next(colors)
                plt.errorbar(self.times,m,yerr = s,color = c, label = k)
                if not clean:
                    m = p[k]
                    plt.plot(self.times,m,color = c,linewidth = .5)    
                plt.legend()
                plt.yscale('log')
                plt.title('Phage Abundances over Time')
                plt.ylabel('pfu/g stool')
                plt.xlabel('Time (d)')
                it += 1
            plt.show()
        if plot_pairs:
            jet= plt.get_cmap('jet')
            colors = iter(jet(np.linspace(0,1,10)))
            for bb,pp in self.pairs:
                fig1, ax1 = plt.subplots()
                # if bb == self.pairs[-1][0]:
                #     import pdb; pdb.set_trace()
                bk = [key for key,val in self.bacteria.items() if bb in key][0]
                
                pk = [key for key,val in p.items() if pp in key][0]
                # import pdb; pdb.set_trace()
                if not clean:
                    ax1.plot(self.times, b[bk],color = 'r', linewidth = .5)
                # import pdb; pdb.set_trace()
                ax1.errorbar(self.times, np.nanmean(b[bk],1),yerr = np.nanstd(b[bk],1),color = 'r',label = bb)
                ax1.tick_params(axis='y', labelcolor='r')
                ax1.set_ylabel(bb + ' Bacteria')
                plt.yscale('log')
                # ax1.legend()
                ax2 = ax1.twinx()
                if not clean:
                    ax2.plot(self.times, p[pk],color = 'g', linewidth = .5)
                ax2.errorbar(self.times, np.nanmean(p[pk],1),yerr = np.nanstd(p[pk],1),color = 'g',label = pp)
                ax2.tick_params(axis='y', labelcolor='g')
                ax2.set_ylabel(pp + ' Phage')
                # ax2.legend()
                plt.title(pp + ' Phage vs ' + bb + ' Bacteria')
                plt.show()
            for bb,pp in self.pairs:
                fig1, ax1 = plt.subplots()
                # if bb == self.pairs[-1][0]:
                #     import pdb; pdb.set_trace()
                bk = [key for key,val in self.bacteria.items() if bb in key][0]
                
                pk = [key for key,val in p.items() if pp in key][0]
                # import pdb; pdb.set_trace()
                if not clean:
                    ax1.plot(self.times, b[bk],color = 'r', linewidth = .5)
                # import pdb; pdb.set_trace()
                ax1.errorbar(self.times, np.nanmean(b[bk],1),yerr = np.nanstd(b[bk],1),color = 'r',label = bb)
                ax1.tick_params(axis='y', labelcolor='r')
                # ax1.set_ylabel(bb + ' Bacteria')
                plt.yscale('log')
                # ax1.legend()
                # ax2 = ax1.twinx()
                if not clean:
                    ax1.plot(self.times, p[pk],color = 'g', linewidth = .5)
                ax1.errorbar(self.times, np.nanmean(p[pk],1),yerr = np.nanstd(p[pk],1),color = 'g',label = pp)
                ax1.tick_params(axis='y', labelcolor='g')
                ax1.set_ylabel('cfu/stool or pfu/stool')
                ax1.legend()
                plt.title(pp + ' Phage vs ' + bb + ' Bacteria')
                plt.savefig(pp+bb + '.png')
                plt.show()


        
        # if m == None:
        #     m = list(self.bacteria.values())[0].columns  
        # if b == None:
        #     b = self.bacteria.keys()
        # if p == None:
        #     p = self.phages.keys()

        

        

        # if b == 'mean':
        #     b = np.nanmean(list(self.bacteria.values())[0].columns)