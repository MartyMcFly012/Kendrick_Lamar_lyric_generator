import string
from tensorflow.keras.preprocessing.text import Tokenizer

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text.lower()

def preprocess_data(text):
    # Remove punctuation and newline character
    punctuation = string.punctuation.replace('\n', '')
    text = text.translate(str.maketrans('', '', punctuation))
    # Split text into individual words
    words = text.split()
    return words

file_path = "Kendrick_Lamar_lyrics.txt"
text = load_data(file_path)
words = preprocess_data(text)

# Creating a dictionary to store words with numerical representations
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(words)

input_sequences = []
target_sequences = []
for i in range(sequence_length, len(words)):
    input_sequences.append(words[i - sequence_length:i])
    target_sequences.append(words[i])

input_sequences = tokenizer.texts_to_sequences(input_sequences)
target_sequences = tokenizer.texts_to_sequences(target_sequences)

# Preparing training data
import numpy as np

input_sequences = np.array(input_sequences)
target_sequences = np.array(target_sequences)

vocab_size = len(tokenizer.word_index) + 1
input_data = input_sequences[:, :-1]
target_data = input_sequences[:, -1]
target_data = np.expand_dims(target_data, axis=1)

# Split the data into training and validation sets
validation_split = 0.2
num_samples = len(input_data)
split_index = int((1 - validation_split) * num_samples)

train_input = input_data[:split_index]
train_target = target_data[:split_index]
val_input = input_data[split_index:]
val_target = target_data[split_index:]

# Building and training the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

embedding_dim = 128
hidden_units = 256

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=sequence_length-1))
model.add(LSTM(hidden_units))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

batch_size = 128
epochs = 50

model.fit(train_input, train_target, batch_size=batch_size, epochs=epochs, validation_data=(val_input, val_target))
# Step 8: Generate Rap Lyrics

# After training the model, you can use it to generate rap lyrics.
# Provide a seed text to start the generation process.
# Use a loop to predict the next word based on the previous words and update the seed text.

def generate_lyrics(seed_text, num_words):
    for _ in range(num_words):
        seed_input = tokenizer.texts_to_sequences([seed_text])[0]
        seed_input = np.array(seed_input)
        predicted_index = np.argmax(model.predict(seed_input), axis=-1)[0]
        predicted_word = tokenizer.index_word[predicted_index]
        seed_text += " " + predicted_word
    return seed_text

seed_text = "I got the power"
generated_lyrics = generate_lyrics(seed_text, num_words=50)
print(generated_lyrics)
