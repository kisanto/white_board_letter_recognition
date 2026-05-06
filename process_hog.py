import cv2
import numpy as np
import os


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

BASE_PATH ="dataset"

size = 22 

hog = cv2.HOGDescriptor(_winSize =(20, 20),_blockSize =(10,10),_blockStride=(5,5), _cellSize=(5,5), _nbins=9)

data=[]
labels = []

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
        img_deskewed = deskew(img_resized)
        hog_features = hog.compute(img_deskewed)
        flattend = hog_features.flatten()
        data.append(flattend)
        labels.append(label_map[letter])

data = np.array(data,dtype=np.float32)
labels = np.array(labels,dtype=np.int32)


np.savez("processed_data_hog.npz", train_data=data,train_labels= labels)

print("\n---etoimos ---")
print(f"oikones poy epeksergastikan: {len(data)}")
print(f"sxhma : {data.shape}") 
print("αποθηκευμενα στο 'processed_data_hog.npz'")