# First Aid Chatbot

A simple chatbot that provides basic first aid advice using a trained deep learning model and Streamlit for the UI.

---

## Installation

Follow these steps to set up the chatbot locally:

1. Clone the repository

```bash
git clone https://github.com/naki-san/Firstaid_Chatbot.git
cd firstaid-chatbot

```

2. Create and activate a virtual environment

```bash
  python -m venv .venv
  .\.venv\Scripts\activate

```
3. Install dependencies

```bash
pip install -r requirements.txt

```
4. Download NLTK data

```bash
python
import nltk
nltk.download('punkt')
exit()

```
5. Run the training script

```bash
python train.py

```

6. Run the chatbot app:

```bash
streamlit run app.py

```
Notes:
Make sure to activate your virtual environment before installing packages or running scripts.

If you add new intents or update intents.json, re-run train.py to retrain the model.




