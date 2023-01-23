import os
import random

path = 'UTKFace/test/adults'
dest = 'BackupData_test'

files = os.listdir(path)
random.shuffle(files)

for file_name in files[:len(files)//2]:
    file_path = os.path.join(path, file_name)
    dest_path = os.path.join(dest, file_name)
    os.rename(file_path, dest_path)
    print(f'{file_name} moved to {dest}')
