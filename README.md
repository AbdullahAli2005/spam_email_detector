# Spam Email Detector

A machine learning–based spam email detection system built using Python and scikit-learn. This project uses the **UCI Spambase Dataset**, which consists of pre-extracted numerical features representing word and character frequencies in emails.

---

## 📌 Project Overview

This project trains a **Naive Bayes classifier** to distinguish between **spam** and **non-spam (ham)** emails. Unlike text-based NLP pipelines, this dataset is already feature-engineered, allowing the model to work directly with numeric inputs.

* **Algorithm:** Multinomial Naive Bayes
* **Dataset:** UCI Spambase
* **Task:** Binary classification (Spam / Not Spam)

---

## 🗂 Dataset Information

* Each row represents an email
* Columns represent:

  * Word frequency features (`word_freq_*`)
  * Character frequency features (`char_freq_*`)
  * Capital letter run-length statistics
* Target column:

  * `spam` → `1 = Spam`, `0 = Not Spam`

No additional text preprocessing or vectorization is required.

---

## 🛠 Technologies Used

* Python 3
* Pandas
* NumPy
* Scikit-learn

---

## 🚀 Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/AbdullahAli2005/spam-email-detector.git
   cd spam-email-detector
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-learn
   ```

---

## ▶️ How to Run

Make sure `spam.csv` is present in the project root directory.

```bash
python main.py
```

---

## 📊 Output

The script outputs:

* Model accuracy
* Confusion matrix

Example:

```text
Accuracy: 89.45%
Confusion Matrix:
[[700  45]
 [ 65 412]]
```

---

## 📈 Model Workflow

1. Load dataset
2. Separate features and labels
3. Split data into training and test sets
4. Train Multinomial Naive Bayes classifier
5. Evaluate using accuracy and confusion matrix

---

## 📁 Project Structure

```
spam_email_detector/
│── main.py
│── spam.csv
│── README.md
│── .venv/
```

---

## 🧠 Future Improvements

* Add Logistic Regression and SVM for comparison
* Perform feature selection
* Save trained model using `joblib`
* Add ROC-AUC evaluation
* Build a simple web interface (Flask / FastAPI)

---

## 📜 License

This project is for educational purposes.

---

## 👤 Author

**Abdullah Ali**

Flutter & Full-Stack Developer

---

Feel free to fork this repository, experiment with different models, and improve performance.
