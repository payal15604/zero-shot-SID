## **Step 1: Compute β from Dataset**
Run the following command to process the dataset and generate `beta_values.csv`:

   `python compute_beta.py`

## **Step 2: Train CNN Model**
Train the CNN to estimate **β** directly from hazy images:

   `python cnn_beta_estimator.py`

## **Step 3: Predict β for a New Hazy Image**

### **Load the trained model:**
   
   `model = BetaCNN().cuda()`
   `model.load_state_dict(torch.load("beta_cnn.pth"))`
   `model.eval()`

### **Preprocess the input image:**
   
   `image = cv2.imread("hazy_image.jpg", cv2.IMREAD_COLOR)`
   `image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`
   `image = np.array(image, dtype=np.float32) / 255.0`
   `image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).cuda()`

### **Predict β:**
   
   `beta_pred = model(image).item()`
   `print("Estimated β:", beta_pred)`
