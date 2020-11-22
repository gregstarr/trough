"""
https://gist.github.com/nasrulhazim/cfd5f01e3b261b09d54f721cc1a7c50d

to download madrigal:
    - open C:\\Users\\Greg\\miniconda3\\envs\\trough\\Lib\\site-packages\\madrigalWeb as pycharm project
    - get command from http://cedar.openmadrigal.org/downloadAsIsScript/
    - run configuration for globalDownload.py
"""
from ftplib import FTP
import datetime
import os

ftp = FTP('swarm-diss.eo.esa.int', timeout=60)
ftp.login()

# base_download_path = "/Advanced/Plasma_Data/2Hz_TII_Cross-track_Dataset/New_baseline/Sat_{}"  # ion drift
base_download_path = "/Advanced/Plasma_Data/2_Hz_Langmuir_Probe_Extended_Dataset/Sat_{}"  # electron density
all_files = {sat: ftp.nlst(base_download_path.format(sat)) for sat in ['A', 'B', 'C']}

date_string = '201'
for sat, files in all_files.items():
    sat_download_path = base_download_path.format(sat)
    for file in files:
        if file[19:19+len(date_string)] != date_string or file[35:35+len(date_string)] != date_string:
            continue
        print(datetime.datetime.now(), file)
        save_path = f"E:\\swarm\\raw\\{file}"
        if os.path.exists(save_path):
            continue
        try:
            with open(save_path, 'wb') as f:
                ftp.retrbinary("RETR " + sat_download_path + '/' + file, f.write)
        except:
            print(f"Couldn't download file: {file}")
ftp.close()
