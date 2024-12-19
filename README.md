# AI Disease Prediction System

This repository contains an AI-powered disease prediction system that utilizes a BERT-based local AI model and GPT-4 for enhanced accuracy. The system predicts diseases based on user-provided symptoms and asks follow-up questions to refine its predictions.

---

## **Features**

1. Local AI model trained on a curated dataset of symptoms and diseases.
2. GPT-4 integration for follow-up questions and diagnosis refinement.
3. User-friendly interface to input symptoms and receive predictions.
4. Continuous learning: Saves new data for future retraining.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/amaannizam29/AI-Disease-Prediction-System
cd your-repo-folder
```

### **2. Install Dependencies**
Make sure you have Python installed (>= 3.7), then run:
```bash
pip install -r requirements.txt
```

### **3. Prepare the AI Model**
Before using the system, you need to set up and train the AI model:

#### **Step 1: Train the Initial Model**
Run the following script to train the AI model using the provided datasets:
```bash
python train_initial_model.py
```
This script will:
- Combine all datasets.
- Train the local AI model on the combined data.
- Save the trained model for predictions.

#### **Step 2: Verify the Training**
Ensure the following files are generated after training:
- `saved_model/` (contains the trained model files).
- `refined_data.csv` (stores refined data for future retraining).

If these files are missing, re-run the training script.

---

## **Using the Prediction System**
Once the model is trained, run the prediction system:
```bash
python predict_disease.py
```
### **How It Works:**
1. Input your symptoms.
2. Local AI predicts the disease.
3. GPT-4 asks follow-up questions to refine the diagnosis.
4. Final predictions from Local AI and GPT-4 are displayed.

---

## **Retraining the Model**
To retrain the model with new data, use:
```bash
python retrain_full_model.py
```
This script updates the model with refined data saved during previous predictions.

---

## **Dataset Requirements**
If you want to use your own dataset, ensure it includes:
- **Symptoms**: A description of symptoms.
- **Disease**: The corresponding disease name.

Place the dataset in `.csv` format in the project folder and retrain the model.

---

## **API Key Setup**
This project requires an OpenAI API key for GPT-4 integration. Set your API key in the script where indicated:
```python
openai.api_key = "your-api-key-here"
```

---

## **Folder Structure**
```plaintext
├── datasets/                # Contains all CSV datasets
├── model/                   # Trained model files
│   ├── saved_model/         # Saved model after training
├── results/                 # Refined data and logs
├── scripts/                 # Python scripts
│   ├── predict_disease.py
│   ├── retrain_full_model.py
│   ├── train_initial_model.py
├── requirements.txt         # Dependencies
├── README.md                # Project description
```

---

## **Contributing**
Contributions are welcome! Feel free to open issues or submit pull requests.

---

## **License**
This project is licensed under the MIT License.
