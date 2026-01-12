import detect_heds_module_path
from holoeye import slmdisplaysdk
import numpy as np
import cv2
import time
from PySpin import PySpin  
import sys
import os

output_folder = r"C:\\Users\\ENGB52\\Desktop\\mplc_soc\\active_region_code\\output_12_11_2025_stripe_width_10_exp100_filter"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

slm_res = [1080, 1920]    # SLM resolution: (height, width)
stripe_width = 10       # Width (in pixels) of the white stripe
delay_time = 0.3            # Delay (in seconds) to allow the SLM to update

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
    camera.Gain.SetValue(0)
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
baseline_pattern = np.zeros((slm_res[0], slm_res[1]), dtype=np.uint8)
slm.showData(baseline_pattern)
time.sleep(delay_time)  # Allow the SLM to update
base_img = capture_image()
if base_img is None:
    print("Baseline capture failed.")
    sys.exit(1)

print("Select ROI from the baseline image. Press ENTER or SPACE after selection, then ESC.")
roi = cv2.selectROI("Select ROI", base_img, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select ROI")
if roi == (0,0,0,0):
    print("No ROI selected. Exiting.")
    sys.exit(1)
print("Selected ROI (x, y, w, h):", roi)

cropped_baseline = base_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cropped_baseline_path = os.path.join(output_folder, "baseline.png")
cv2.imwrite(cropped_baseline_path, cropped_baseline)
print(f"Cropped baseline image saved as '{cropped_baseline_path}'.")
cv2.imshow("Cropped Baseline", cropped_baseline)
cv2.waitKey(500)
cv2.destroyAllWindows()

num_stripes = slm_res[1] // stripe_width  # Number of stripes that fit in the SLM width
#num_stripes = slm_res[0] // stripe_width  # Number of stripes that fit in the SLM height

for i in range(num_stripes):
    pos = i * stripe_width
    print(f"\n--- Processing stripe {i} at position {pos} ---")

    stripe_pattern = np.zeros((slm_res[0], slm_res[1]), dtype=np.uint8)
    stripe_pattern[:, pos:pos+stripe_width] = 255
    #stripe_pattern[pos:pos+stripe_width,:] = 255
    slm.showData(stripe_pattern)
    time.sleep(delay_time) 

    stripe_img = capture_image()
    if stripe_img is None:
        print(f"Skipping stripe {i} due to stripe capture error.")
        continue
    
    x, y, w, h = roi
    cropped_stripe = stripe_img[int(y):int(y+h), int(x):int(x+w)]
    cropped_stripe_path = os.path.join(output_folder, f"cropped_stripe_{i}.png")
    cv2.imwrite(cropped_stripe_path, cropped_stripe)
    print(f"Cropped stripe {i} saved as '{cropped_stripe_path}'.")
    # cv2.imshow(f"Cropped Stripe {i}", cropped_stripe)
    # cv2.waitKey(500)
    # cv2.destroyAllWindows()

slm.close()
slm = None
print("Process completed. All resources have been released.")

