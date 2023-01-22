import os

path = 'UTKFace/train/adults'

i=0
for file_name in os.listdir(path):
    if file_name.endswith('.jpg'):
        age = int(file_name.split('_')[0])
        if False: #age > 85:
            file_path = os.path.join(path, file_name)
            os.remove(file_path)
            print(f'{file_name} deleted')
        i+=1
print(f'deleted {i} images')