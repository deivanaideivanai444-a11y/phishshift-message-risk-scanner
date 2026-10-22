# 🛡️ PhishShift Message Risk Scanner

### ML-Based Phishing Message Detection Using NLP

PhishShift Message Risk Scanner is a cybersecurity and machine learning project designed to analyse simulated SMS and email messages and identify potential phishing risks.

The project compares rule-based detection with machine learning models and explains suspicious message characteristics to support safer decisions.

## 🚀 Features

- 📩 Analyse SMS and email message text.
- 🛡️ Identify potential phishing indicators.
- 🤖 Compare rule-based and machine learning approaches.
- 📊 Evaluate models using standard classification metrics.
- 🔎 Display suspicious risk cues and explanations.
- ⚠️ Support uncertain predictions and manual review.
- 🧪 Test model robustness against harmless text variations.
- 📱 Responsive web interface built with Flask, HTML, CSS and JavaScript.

## 🧰 Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas and NumPy
- TF-IDF
- Logistic Regression
- Linear Support Vector Machine (SVM)
- HTML, CSS and JavaScript
- Matplotlib and Seaborn

## 📁 Project Structure

```text
phishshift-message-risk-scanner/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── train_model.py
├── evaluate_model.py
├── generate_dataset.py
├── data/
├── docs/
├── models/
├── reports/
├── results/
├── src/
├── static/
├── templates/
└── tests/
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/deivanaideivanai444-a11y/phishshift-message-risk-scanner.git
cd phishshift-message-risk-scanner
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Open this address in your browser:

http://127.0.0.1:5000

Keep the terminal running while using the application.

## 🧠 Train and Evaluate Models

Generate the synthetic dataset:

```bash
python generate_dataset.py
```

Train the models:

```bash
python train_model.py
```

Evaluate model performance:

```bash
python evaluate_model.py
```

Run tests:

```bash
python run_tests.py
```

Run these commands in the documented order and check the actual output. Existing saved models or results may be overwritten depending on the implementation.

## 📊 Evaluation

The project is designed to evaluate:

- Accuracy, precision, recall and F1-score.
- Confusion matrices.
- False positives on legitimate urgent messages.
- Generalisation to unseen campaign groups.
- Robustness to harmless text changes.
- False positives and false negatives.

Actual results should be taken from the generated evaluation files. Model performance is not guaranteed.

## 🔐 Security and Privacy

- Uses synthetic training examples for educational evaluation.
- Does not require real passwords, OTPs or private mailbox data.
- Submitted URLs should not be opened or fetched.
- Suspicious messages should be verified through official channels.
- Predictions are decision-support signals, not proof that a message is safe or malicious.

## ⚠️ Limitations

- Predictions depend on the quality and diversity of the training data.
- New phishing techniques may evade detection.
- Legitimate urgent messages may be incorrectly flagged.
- Confidence scores do not guarantee prediction correctness.
- Independent testing on unseen campaigns is necessary.

## 🎯 Project Objective

To build and evaluate an educational phishing-risk scanner that combines NLP, machine learning, risk-cue explanations and campaign-separated evaluation.

## 👩‍💻 Author

**Deivanai G.**

B.Tech — Artificial Intelligence and Data Science

## 📄 Disclaimer

This project is intended for educational and cybersecurity awareness purposes only. It is not a replacement for professional security tools or human verification.

## ⭐ Acknowledgement

This project was developed as part of a cybersecurity and machine learning learning exercise. Dataset sources, licensing details, and experimental evidence should be documented in the repository.

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
