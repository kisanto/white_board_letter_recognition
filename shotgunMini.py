import cv2
import os
import numpy as np

# --- CONFIGURATION ---
BASE_PATH = "dataset1.3"
# We will save the cleaned versions in a new folder to avoid messing up the originals
OUTPUT_PATH = "dataset1.3.1" 

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Get all letter folders (A, B, C...)
letters = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]

for letter in letters:
    input_folder = os.path.join(BASE_PATH, letter)
    output_folder = os.path.join(OUTPUT_PATH, letter)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"Cleaning folder: {letter}...")
    
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        
        # 1. Load the image
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 2. The Pre-processing Pipeline
        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur to remove noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive Threshold (The most important part for whiteboards)
        # This makes the ink white and background black
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 3)
        # The '_,' catches the extra return value so 'thresh' stays an image
        #_, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

        # Now erosion will work perfectly
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        # 3. Save the result
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, thresh)

print("\n--- DONE! ---")
print(f"All cleaned images are in the '{OUTPUT_PATH}' folder.")