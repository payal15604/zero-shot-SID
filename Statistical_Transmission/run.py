import cv2
import numpy as np
from bounding_fun import bounding_function

# Load an image
I = cv2.imread("/home/student1/Desktop/Zero_Shot/Datasets/SOTS/indoor/hazy/train/1436_9.png")  
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)  

# Process the image
zeta = 1  # Example parameter value
r, est_tr_proposed, A = bounding_function(I, zeta)

# Convert result to 8-bit format
output_image = (r * 255).astype(np.uint8)

# Resize image for smaller display
display_width = 600  # Adjust as needed
display_height = 400  # Adjust as needed
resized_image = cv2.resize(output_image, (display_width, display_height))

# Save and Display Results
cv2.imwrite("output.jpg", output_image)
cv2.imshow("Dehazed Image", resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

