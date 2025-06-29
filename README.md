# 🤖 Rule-Based Chatbot with Seq2Seq (Small Conversation)

This project implements a **simple conversational chatbot** combining **rule-based responses** with a **Seq2Seq neural network model** for small conversations. It demonstrates fundamental chatbot development techniques, combining deterministic rules with machine learning-based natural language understanding.



## 📑 Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Project Structure](#project-structure)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



## 📝 Introduction

**Chatbots** are conversational agents designed to interact with users in natural language. This project combines:

✅ **Rule-Based Chatbot** – Uses predefined patterns to respond to common greetings or questions

✅ **Seq2Seq Model** – Trained on small conversation datasets for generative responses when rules do not match

This hybrid approach ensures the chatbot provides reliable, quick responses while also handling open-ended queries with learned Seq2Seq responses.



## ✨ Features

✔️ **Pattern Matching** – Uses regex or keyword-based matching for rule-based responses

✔️ **Fallback Seq2Seq Responses** – Uses a trained encoder-decoder neural network model when no rule matches

✔️ **Preprocessing Pipeline** – Tokenization, padding, and text cleaning for neural model inputs

✔️ **Small Conversation Dataset Training** – Trained on simple question-answer pairs

✔️ **Interactive Chat Interface** – Terminal-based conversation with user input



## 🛠️ Technologies Used

* **Python 3**
* **TensorFlow / Keras**
* `numpy`
* `pandas`
* `re` (regular expressions for rule-based logic)
* **Jupyter Notebook**



## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Rule-Based-Chatbot-with-Seq2Seq-Small-Conversation.git
cd Rule-Based-Chatbot-with-Seq2Seq-Small-Conversation
```

2. **Create and activate a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```



## ▶️ Usage

1. Open `Rule_Based_Seq2Seq_Chatbot.ipynb` in Jupyter Notebook.
2. Run cells sequentially to:

   * Load and preprocess conversation dataset
   * Build and train the Seq2Seq model
   * Define rule-based patterns and responses
   * Run the interactive chatbot loop in terminal or notebook cell



## 🏗️ Model Architecture

### **Seq2Seq Neural Network**

* **Encoder:** Embedding + LSTM layers to encode input sequence into context vector
* **Decoder:** Embedding + LSTM + Dense with softmax activation to generate response sequence

Sample architecture snippet:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Encoder
encoder_inputs = Input(shape=(None,))
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(units, return_state=True)(x)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
```



## 📁 Project Structure

```
Rule-Based-Chatbot-with-Seq2Seq-Small-Conversation/
 ┣ data/
 ┃ ┗ conversations.csv
 ┣ Rule_Based_Seq2Seq_Chatbot.ipynb
 ┣ requirements.txt
 ┗ README.md
```



## 📈 Results

* **Rule-Based Accuracy:** Immediate response for matched patterns (greetings, farewells, FAQs)
* **Seq2Seq Model:** Generates simple but meaningful responses for open-ended inputs

Example:

```
User: Hi
Bot: Hello! How can I assist you today?

User: What is your name?
Bot: I am your AI chatbot assistant.

User: Tell me about AI
Bot: AI is the simulation of human intelligence processes by machines.
```



## 🤝 Contributing

Contributions are welcome to:

* Expand rule-based responses for broader coverage
* Train Seq2Seq model on larger conversational datasets
* Integrate with Flask or FastAPI for web deployment
* Build a Flutter frontend for mobile integration

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request



## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



## 📬 Contact

**Ugama Benedicta Kelechi**
[LinkedIn](www.linkedin.com/in/ugama-benedicta-kelechi-codergirl-103041300) | [Email](mailto:ugamakelechi501@gmail.com) | [Your Portfolio](#)



### ⭐️ If you find this project useful, please give it a star!

