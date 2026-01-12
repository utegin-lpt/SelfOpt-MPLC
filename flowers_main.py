import detect_heds_module_path
from holoeye import slmdisplaysdk
import numpy as np
import cv2
import time
from PySpin import PySpin  
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifier
from flowers_soc import *
from skimage.measure import block_reduce
import random
from sklearn.model_selection import train_test_split


directory = r"C:/Users/ENGB52/Desktop/mplc_soc/flowers17_parts_200x200"


SLM_RES = (1080, 1920)
DELAY_TIME = 0.3
EMBED_HEIGHT = 200
EMBED_COUNT = 3  # set 3 for RGB, 1 for Grayscale
RUNS = [(120, 320), (550, 750), (940, 1140), (1300, 1500)]


def create_composite(background_shape, runs, base_img, mat, embed_count=1, embed_height=200):
    H, W = background_shape
    composite = np.zeros((H, W), dtype=np.uint8)
    #mat = block_reduce(mat,(1,2,2),func=np.mean)
    mat = np.kron(mat, np.ones((1,4,4)))
    mat = np.array(mat, dtype=np.uint8)
    if embed_count==3:
        
        total_embed_height = embed_count * embed_height + 40
        if total_embed_height > H:
            raise ValueError(f"Total embed height ({total_embed_height}) exceeds background height ({H}).")
        
        vertical_offset = (H - total_embed_height) // 2
        
        base_img = np.array(base_img, dtype=np.uint8)
        #print(base_img.shape)
        #print(np.max(base_img))
        R = base_img[:,:,0]
        G = base_img[:,:,1]
        B = base_img[:,:,2]
        
        composite[vertical_offset: vertical_offset + embed_height, runs[0][0]:runs[0][0]+200] = R
        composite[vertical_offset + embed_height + 20: vertical_offset + 2*embed_height + 20, runs[0][0]:runs[0][0]+200] = G
        composite[vertical_offset + 2*embed_height + 40: vertical_offset + 3*embed_height + 40, runs[0][0]:runs[0][0]+200] = B
        
        for i in range(1,4):
            composite[vertical_offset: vertical_offset + embed_height, runs[i][0]:runs[i][0]+200] = mat[(i-1)*3]
            composite[vertical_offset + embed_height + 20: vertical_offset + 2*embed_height + 20, runs[i][0]:runs[i][0]+200] = mat[(i-1)*3 +1]
            composite[vertical_offset + 2*embed_height + 40: vertical_offset + 3*embed_height + 40, runs[i][0]:runs[i][0]+200] = mat[(i-1)*3 +2]
    return composite

def init_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        print("No FLIR cameras detected.")
        cam_list.Clear()
        system.ReleaseInstance()
        sys.exit(1)
    camera = cam_list[0]
    camera.Init()
    camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    camera.ExposureTime.SetValue(100.0)  # Adjust exposure as needed
    camera.GainAuto.SetValue(PySpin.GainAuto_Off)
    camera.Gain.SetValue(0.0)
    camera.BeginAcquisition()
    return camera, cam_list, system

def cleanup_camera(camera, cam_list, system):
    camera.EndAcquisition()
    camera.DeInit()
    del camera
    cam_list.Clear()
    #system.ReleaseInstance()

def capture_image():
    camera, cam_list, system = init_camera()
    image_result = camera.GetNextImage(1000)
    if image_result.IsIncomplete():
        print("Captured image is incomplete. Status:", image_result.GetImageStatus())
        image_result.Release()
        cleanup_camera(camera, cam_list, system)
        return None
    img = image_result.GetNDArray()
    image_result.Release()
    cleanup_camera(camera, cam_list, system)
    del camera, cam_list, system
    return cv2.flip(img, 1)

def shuffle_data(directory):
    data = []
    class_names = sorted(os.listdir(directory))

    for class_name in class_names:
        class_folder = os.path.join(directory, class_name)
        if not os.path.isdir(class_folder):
            continue

        #for img_file in tqdm(os.listdir(class_folder), desc=f"Loading {class_name}"):
        for img_file in os.listdir(class_folder):
            if img_file.lower().endswith(".jpg"):
                img_path = os.path.join(class_folder, img_file)
                data.append((img_path,class_name))             
    return data
    
def run_experiment(slm, data, mat, N=1, out_directory='', save=False, plot=False):
    
    mat = np.array([
        mat[i*50:(i+1)*50, j*50:(j+1)*50]
        for i in range(3)
        for j in range(3)
    ])
    
    data = np.concatenate([data[i] for i in range(len(data))], axis=0)
    out_imgs = []
    labels = []
    
    print(f"Running experiment with N={N} frame averaging...")
    
    for idx, (img_path, class_name) in enumerate(data):
        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)
        
        composite = create_composite(SLM_RES, RUNS, img, mat, embed_count=EMBED_COUNT)
        
        if plot: 
            plt.imshow(img)
            plt.title(f"Input image: {class_name}")
            plt.show()
            plt.imshow(composite, cmap='gray')
            plt.title("SLM pattern")
            plt.show()
            
        slm.showData(composite)
        time.sleep(DELAY_TIME)
        
        # ====================================================================
        # FRAME AVERAGING
        # ====================================================================
        if N == 1:
            out_img = capture_image()
        else:
            frames = []
            for frame_idx in range(N):
                frame = capture_image()
                if frame is None:
                    print(f"Warning: Frame {frame_idx+1}/{N} capture failed for {img_path}")
                    continue
                frames.append(frame.astype(np.float64))  
                time.sleep(0.01) 
            
            if len(frames) == 0:
                print(f"ERROR: All frame captures failed for {img_path}")
                continue
            elif len(frames) < N:
                print(f"Warning: Only {len(frames)}/{N} frames captured successfully")
            
            frames_stack = np.stack(frames, axis=0)  
            out_img = np.mean(frames_stack, axis=0) 
            
            out_img = out_img.astype(np.uint8)
        
        # ====================================================================
        # Crop to ROI
        # ====================================================================
        out_img = out_img[40:490, 300:450]
        out_imgs.append(out_img)
        labels.append(class_name)
        
        # Check for saturation
        if np.max(out_img) >= 255:
            print(f"SATURATION detected in image {idx+1}/{len(data)} ({class_name})")

        if plot:
            plt.imshow(out_img, cmap='gray')
            plt.title(f"Output: {class_name}")
            plt.colorbar()
            plt.show()
    
    labels = np.array(labels)
    out_imgs = np.array(out_imgs).astype(np.float32)
    
    return out_imgs, labels


def run_experiment_(slm, data_train, mat, indx, N_frames):
    data_train_subsub = [data_train[i] for i in indx]
    out_imgs_tr, labels_tr = run_experiment(slm, data_train_subsub, mat, N=N_frames)
    
    return out_imgs_tr, labels_tr


def ridge_classifier_acc(data, labels):

    X_train, X_test, y_train, y_test = train_test_split(
        data,labels, test_size=0.2, random_state=42, stratify=labels
    )
   
    y_train = np.array(y_train) 
    y_test = np.array(y_test) 
    X_train = block_reduce(X_train, block_size=(1,6,6), func=np.mean) / 255.0
    X_test = block_reduce(X_test, block_size=(1,6,6), func=np.mean) / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1) 
    X_test = X_test.reshape(X_test.shape[0], -1) 

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)


    clf = RidgeClassifier()
    clf.fit(X_train, y_train)
    # Evaluate
    accuracy = clf.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.3f}")
    return accuracy

#  Initialize the SLM 
slm = slmdisplaysdk.SLMInstance()
error = slm.open()
if error:
    print("Error opening SLM:", error)
    sys.exit(1)
print("SLM opened successfully.")

# Capture baseline black image 
baseline_pattern = np.zeros((SLM_RES[0], SLM_RES[1]), dtype=np.uint8)
slm.showData(baseline_pattern)
time.sleep(DELAY_TIME) 
base_img = capture_image()
if base_img is None:
    print("Baseline capture failed.")
    sys.exit(1)
cv2.imshow("Baseline", base_img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# ========================================================================
# RUN OPTIMIZATION
# ========================================================================

np.random.seed(42)
random.seed(42)
phase_mask = np.random.randint(0, 256, (9, 50, 50), dtype=np.uint8)
phase_mask = np.block([
    [phase_mask[0], phase_mask[1], phase_mask[2]],
    [phase_mask[3], phase_mask[4], phase_mask[5]],
    [phase_mask[6], phase_mask[7], phase_mask[8]]
 ])


data = []
for p in ["part1","part2"]:
    part_path = os.path.join(directory, p)
    data.append(shuffle_data(part_path))

results = soc_optimize_experimental(
    slm, data,
     measure_accuracy_fn=ridge_classifier_acc,
     upload_masks_fn=run_experiment_,
     initial_mask=phase_mask,
     mask_size=(150,150),
     max_iterations=100,
     threshold=4,
     grains_per_neighbor=1,
     perturbation_scale=15,  # uint8 space (0-255)
     verbose=True,
     save_every=25
 )



save_masks_uint8(results['best_mask'], results['best_accuracy'], 
                 results['accuracy_best_mask'],
                 results['accuracy_history'],
                 results['avalanche_size_history'],
                 results['initial_mask'],
                 results['mask_history'],
                 'best_masks_final_18_11_25_night.npz')
plot_optimization_results(results)

print("\n" + "="*70)
print(f"OPTIMIZATION COMPLETE")
print(f"Best accuracy: {results['best_accuracy']:.4f}")
print("Masks saved as uint8 arrays (0-255)")
print("="*70)
 
slm.close()
slm = None
print("Process completed.")
