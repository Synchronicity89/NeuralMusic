import os

directory = "data"
for filename in os.listdir(directory):
    os.rename(directory+'/'+filename, directory+'/'+filename+'.mid')
