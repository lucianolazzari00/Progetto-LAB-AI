import os
import shutil

# Change this to the path where you extracted the UTKFace dataset
data_dir = 'UTKFace/'

# Create the target directories if they don't exist
os.makedirs(data_dir + 'train/adults', exist_ok=True)
os.makedirs(data_dir + 'train/children', exist_ok=True)
os.makedirs(data_dir + 'val/adults', exist_ok=True)
os.makedirs(data_dir + 'val/children', exist_ok=True)
os.makedirs(data_dir + 'test/adults', exist_ok=True)
os.makedirs(data_dir + 'test/children', exist_ok=True)

# Split the data into 80% train, 10% val, 10% test
for i, filename in enumerate(os.listdir(data_dir)):
    if i % 10 < 8:
        destination = data_dir + 'train/'
    elif i % 10 < 9:
        destination = data_dir + 'val/'
    else:
        destination = data_dir + 'test/'
    
    age, gender, _, _ = filename.split("_")
    age = int(age)
        
    # Move the file to the appropriate folder
    if age < 18:
        shutil.move(data_dir + filename, destination + 'children/' + filename)
    elif age >= 18:
        shutil.move(data_dir + filename, destination + 'adults/' + filename)

