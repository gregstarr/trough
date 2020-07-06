"""
https://gist.github.com/nasrulhazim/cfd5f01e3b261b09d54f721cc1a7c50d
"""
from ftplib import FTP
import georinex as gr
import os

ftp = FTP('swarm-diss.eo.esa.int')
ftp.login()

base_download_path = "/Advanced/Plasma_Data/2Hz_TII_Cross-track_Dataset/New_baseline/Sat_{}"
all_files = {sat: ftp.nlst(base_download_path.format(sat)) for sat in ['A', 'B', 'C']}

for sat, files in all_files.items():
    sat_download_path = base_download_path.format(sat)
    for file in files:
        if file[19:25] != '201810' or file[35:41] != '201810':
            continue
        print(file)
        save_path = f"E:\\swarm\\{file}"
        try:
            with open(save_path, 'wb') as f:
                ftp.retrbinary("RETR " + sat_download_path + '/' + file, f.write)
        except:
            print(f"Couldn't download file: {file}")
ftp.close()
