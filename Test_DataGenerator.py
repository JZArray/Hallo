import DataGenerator as genarator
import os
from load_data import LoadData as LoadData


files = ['RMS的Extraction.csv10-01-2019 15:54:14']
#datapath = 'Extractiondata/RMS_Extraction/'
column = ['energie','entropy','mean']
prediction_columns = ['energie','entropy','mean']
datapath = 'Extractiondata/RMS_Extraction/'
filename = 'RMS的Extraction.csv10-01-2019 15:54:14'
a = genarator.generate_data_all_files(files=files,columns=column,prediction_columns=prediction_columns,prediction_timeteps=2,window_timeteps=4,batch_size=8,type = None,progress_tune=None,LoadData=LoadData,data_path=datapath,csv_path=None,norm=None,normData=False,filetype='csv',verbose=0)

for i in a:
    print('处理中......')