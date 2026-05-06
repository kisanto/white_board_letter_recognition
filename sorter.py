import cv2
import os

# --- CONFIGURATION ---
BASE_PATH = "dataset2"

# Get all letter folders
letters = sorted([f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))])

print("--- DATASET SORTER ---")
print("Instructions: Press 'y' to KEEP, 'n' to DELETE, 'q' to QUIT")

for letter in letters:
    folder_path = os.path.join(BASE_PATH, letter)
    images = os.listdir(folder_path)
    
    print(f"\nSorting letter: {letter} ({len(images)} images)")
    
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        # Show the image (zoomed in so you can see it clearly)
        display_img = cv2.resize(img, (200, 200))
        cv2.imshow("Sort: 'y'=Keep, 'n'=Delete, 'q'=Quit", display_img)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('y'):
            # Keep it, do nothing
            continue
        elif key == ord('n'):
            # Delete the file
            os.remove(img_path)
            print(f"Deleted: {img_name}")
        elif key == ord('q'):
            print("Exiting sorter...")
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("\n--- DONE SORTING ---")