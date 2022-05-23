import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt 
from matplotlib.pyplot import figure


truong_tieu_hoc_dataframe = pd.read_excel('datasrc.xlsx')
ti_le_toan_colum = 'Mathematics Rate'
ti_le_doc_colum = 'Reading Rate'

trung_binh_toan = truong_tieu_hoc_dataframe[ti_le_toan_colum].mean()
lech_chuan_toan = truong_tieu_hoc_dataframe[ti_le_toan_colum].std()

trung_binh_doc = truong_tieu_hoc_dataframe[ti_le_doc_colum].mean()
lech_chuan_doc = truong_tieu_hoc_dataframe[ti_le_doc_colum].std()

toan_min = min(truong_tieu_hoc_dataframe[ti_le_toan_colum])
toan_max = max(truong_tieu_hoc_dataframe[ti_le_toan_colum])

doc_min = min(truong_tieu_hoc_dataframe[ti_le_doc_colum])
doc_max = max(truong_tieu_hoc_dataframe[ti_le_doc_colum])

chuan_hoa_toan = np.nan_to_num((truong_tieu_hoc_dataframe[ti_le_toan_colum] - toan_min)/(toan_max - toan_min))
chuan_hoa_doc = np.nan_to_num((truong_tieu_hoc_dataframe[ti_le_doc_colum] - doc_min)/(doc_max - doc_min))

truong_tieu_hoc_dataframe['chuan_hoa_toan'] = chuan_hoa_toan
truong_tieu_hoc_dataframe['chuan_hoa_doc'] = chuan_hoa_doc


toan_va_doc = np.column_stack((truong_tieu_hoc_dataframe['chuan_hoa_toan'], truong_tieu_hoc_dataframe['chuan_hoa_doc']))

kmean_tieu_hoc = KMeans(n_clusters=2).fit(toan_va_doc)

cac_cum = kmean_tieu_hoc.cluster_centers_

truong_tieu_hoc_dataframe['cac_cum'] = kmean_tieu_hoc.labels_

truong_tieu_hoc_dataframe1 = truong_tieu_hoc_dataframe[truong_tieu_hoc_dataframe['cac_cum'] == 0]
truong_tieu_hoc_dataframe2 = truong_tieu_hoc_dataframe[truong_tieu_hoc_dataframe['cac_cum'] == 1]

figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
plt.xlabel('So Lieu Toan')
plt.ylabel('So Lieu Doc') 
plt.scatter(truong_tieu_hoc_dataframe1.chuan_hoa_toan, truong_tieu_hoc_dataframe1.chuan_hoa_doc, alpha = 0.25, s = 100, color='red')
plt.scatter(truong_tieu_hoc_dataframe2.chuan_hoa_toan, truong_tieu_hoc_dataframe2.chuan_hoa_doc, alpha = 0.25, s = 100, color='green')
plt.scatter(cac_cum[:,0], cac_cum[:,1], s = 100000, alpha=0.30)

plt.show()