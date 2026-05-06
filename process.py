import cv2
import os
import numpy as np

BASE_PATH = "dataset"
size =20

data= []
labels= []

letters = sorted(os.listdir(BASE_PATH))
label_map ={letter:i for i ,letter in enumerate(letters)}
print(f"βρεθηκε{len(letters)} γραμμα: {letters}")

for letter in letters :
    letter_path = os.path.join(BASE_PATH,letter)
    if not os.path.isdir(letter_path): continue
    print(f"επεξεργασια γραμματος :{letter}")
    for img_name in os.listdir(letter_path):
        img_path = os.path.join(letter_path,img_name)
        img =cv2.imread(img_path ,cv2.IMREAD_GRAYSCALE)
        if img is None : continue 
        img_resized = cv2.resize(img,(size,size))
        flattend = img_resized.flatten()
        data.append(flattend)
        labels.append(label_map[letter])
data = np.array(data,dtype=np.float32)
labels = np.array(labels,dtype=np.float32)

np.savez("processed_data.npz", train_data=data,train_labels= labels)

print("\n---etoimos ---")
print(f"oikones poy epeksergastikan: {len(data)}")
print(f"sxhma : {data.shape}") 
print("αποθηκευμενα στο 'processed_data.npz'")