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
import torch
from turbo import Turbo1
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

directory = r"C:\Users\ENGB52\Desktop\MPLC\STL-10-small"

SLM_RES = (1080, 1920)
DELAY_TIME = 0.2
EMBED_HEIGHT = 280
EMBED_COUNT = 3  # set 3 for RGB, 1 for Grayscale
RUNS = [(380, 660), (790, 1070), (1180, 1460), (1560, 1840)]


def create_composite(background_shape, runs, base_img, mat, coeff, embed_count=3, embed_height=280):
    H, W = background_shape
    composite = np.zeros((H, W), dtype=np.uint8)

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
        
        RG = (coeff[0] * R + coeff[1] * G) / (coeff[0] + coeff[1])
        RB = (coeff[2] * R + coeff[3] * B) / (coeff[2] + coeff[3])
        GB = (coeff[4] * G + coeff[5] * B) / (coeff[4] + coeff[5])
        RG = np.array(RG, dtype=np.uint8)
        RB = np.array(RB, dtype=np.uint8)
        GB = np.array(GB, dtype=np.uint8)
        
        composite[vertical_offset: vertical_offset + embed_height, runs[0][0]:runs[0][0]+280] = RG
        composite[vertical_offset + embed_height + 20: vertical_offset + 2*embed_height + 20, runs[0][0]:runs[0][0]+280] = RB
        composite[vertical_offset + 2*embed_height + 40: vertical_offset + 3*embed_height + 40, runs[0][0]:runs[0][0]+280] = GB
        
        for i in range(1,4):
            composite[vertical_offset: vertical_offset + embed_height, runs[i][0]:runs[i][0]+280] = mat[(i-1)*3]
            composite[vertical_offset + embed_height + 20: vertical_offset + 2*embed_height + 20, runs[i][0]:runs[i][0]+280] = mat[(i-1)*3 +1]
            composite[vertical_offset + 2*embed_height + 40: vertical_offset + 3*embed_height + 40, runs[i][0]:runs[i][0]+280] = mat[(i-1)*3 +2]
        
        
        
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
    camera.ExposureTime.SetValue(130.0)  # Adjust exposure as needed
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

#  Initialize the SLM 
slm = slmdisplaysdk.SLMInstance()
error = slm.open()
if error:
    print("Error opening SLM:", error)
    sys.exit(1)
print("SLM opened successfully.")

# Capture baseline black image 
baseline_pattern = np.zeros((SLM_RES[0], SLM_RES[1]), dtype=np.uint8)
img = Image.open(r"C:\Users\ENGB52\Desktop\MPLC\RSSCN7\River\1212.jpg")
img = img.resize((SLM_RES[0], SLM_RES[1]))
img = np.array(img, dtype=np.uint8)
slm.showData(img)
time.sleep(DELAY_TIME) 
base_img = capture_image()
if base_img is None:
    print("Baseline capture failed.")
    sys.exit(1)
cv2.imshow("Baseline", base_img)
cv2.waitKey(1000)
cv2.destroyAllWindows()



np.random.seed(42)
mat_save = np.random.randint(0, 256, (9, 10, 10), dtype=np.uint8)
mat_save = np.kron(mat_save, np.ones((24, 24)))
padding = 20
mat_save = np.pad(mat_save, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)  

def objective(coeff):
    out_imgs = []
    labels = []
    class_names = sorted(os.listdir(directory))

    for class_name in class_names:
        class_folder = os.path.join(directory, class_name)
        if not os.path.isdir(class_folder):
            continue

        for img_file in tqdm(os.listdir(class_folder), desc=f"Loading {class_name}"):
            if img_file.lower().endswith(".jpg"):
                img_path = os.path.join(class_folder, img_file)

            img = Image.open(img_path)
            img = img.resize((240,240))
            padding = (20, 20, 20, 20)  # (left, top, right, bottom)
            img = ImageOps.expand(img, padding, fill=0)
            #plt.imshow(img)
            #plt.show()
  
            composite = create_composite(SLM_RES, RUNS, img, mat_save, coeff, embed_count=EMBED_COUNT)
            #plt.imshow(composite)
            #plt.show()
            slm.showData(composite)
            time.sleep(DELAY_TIME)
            out_img = capture_image()
            out_imgs.append(out_img)
            labels.append(class_name)
                
            if np.max(out_img) == 255:
                print("SATURATION")
            out_img = out_img[:,350:530]
            #plt.imshow(out_img)
            #plt.show()
    out_imgs = np.array(out_imgs)   
    out_imgs = block_reduce(out_imgs, block_size=(1,10,10), func=np.mean)
    out_imgs = out_imgs / 255.0
    out_imgs = out_imgs.reshape(out_imgs.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(
        out_imgs, labels, test_size=0.2, random_state=42, stratify=labels
    )
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = LabelEncoder.fit_transform(y_test)
        
    clf = RidgeClassifier(alpha=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test acc: {acc}")
        
    return acc
        
class TargetFunc:
    def __init__(self, forward_func, dim= 6):
        self.forward_func = forward_func
        self.dim = dim
        self.ub = np.ones(dim) # lower boundary
        self.lb = np.zeros(dim) # upper boundary
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        return -1 *self.forward_func(x)
        
f = TargetFunc(objective)

turbo_m = Turbo1(
    f = f,
    lb = f.lb,
    ub = f.ub,
    n_init = 6*1,
    max_evals = 6*2,
    batch_size = 1,
    verbose = True,
    use_ard = True,
    max_cholesky_size = 2000,
    n_training_steps = 50,
    device = "cpu",
    dtype = "float32",
    )

turbo_m.optimize()

X = turbo_m.X
fX = turbo_m.fX
best_idx = np.argmin(fX)
best_acc, best_matrices = fX[best_idx], X[best_idx, :]
np.save("best_coeff_12_07_2025_stl.npy", best_matrices)

print(f"Best accuracy found is {best_acc}")   

         
slm.close()
slm = None
print("Process completed.")
