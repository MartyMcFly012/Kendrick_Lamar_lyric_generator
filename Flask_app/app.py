from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Step 1 to Step 6: Data preparation and text generation

# Load the Kendrick Lamar lyrics from a file
with open("Kendrick_Lamar_lyrics.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Preprocessing steps

# Step 2: Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Step 3: Sequencing
input_sequences = []
output_words = []
sequence_length = 10

# Generate input-output pairs
tokens = tokenizer.texts_to_sequences([text])[0]

for i in range(sequence_length, len(tokens)):
    input_sequence = tokens[i-sequence_length:i]
    output_word = tokens[i]
    input_sequences.append(input_sequence)
    output_words.append(output_word)

# Convert to numpy arrays
input_sequences = np.array(input_sequences)
output_words = np.array(output_words)

# Step 4: NHMM Model
transition_matrix = np.zeros((total_words, total_words))
for sequence in input_sequences:
    for i in range(1, len(sequence)):
        prev_word_id = sequence[i-1]
        curr_word_id = sequence[i]
        transition_matrix[prev_word_id][curr_word_id] += 1

# Normalize transition matrix
transition_matrix = transition_matrix / \
    transition_matrix.sum(axis=1, keepdims=True)


def sample_next_word(seed_word_id, diversity=1.0):
    next_word_probs = transition_matrix[seed_word_id]
    scaled_probs = np.power(next_word_probs, diversity)
    scaled_probs /= np.sum(scaled_probs)
    next_word_id = np.random.choice(range(total_words), p=scaled_probs)
    return next_word_id


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        seed_text = request.form['seed_text']
        num_words = int(request.form['num_words'])
        diversity = float(request.form['diversity'])

        # Step 6: Text Generation
        seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        generated_sequence = seed_sequence[:]

        for _ in range(num_words):
            next_word_id = sample_next_word(
                generated_sequence[-1], diversity=diversity)
            generated_sequence.append(next_word_id)

        generated_lyrics = tokenizer.sequences_to_texts(
            [generated_sequence])[0]
        generated_lyrics = generated_lyrics.capitalize()

        # Process the generated text to enhance the narrative structure
        sentences = generated_lyrics.split('. ')
        sentences = [sentence.strip()
                     for sentence in sentences if sentence.strip()]

        # Create a narrative structure with introduction, rising action, climax, and resolution
        narrative_text = ""
        if len(sentences) >= 4:
            introduction = sentences[0] + ". " + sentences[1] + ". "
            rising_action = ". ".join(sentences[2:-2]) + ". "
            climax = sentences[-2] + ". "
            resolution = sentences[-1] + ". "

            narrative_text = introduction + rising_action + climax + resolution
        else:
            narrative_text = generated_lyrics

        # Reflect widening perspective and expanding consciousness
        narrative_lines = narrative_text.split(". ")
        expanded_lines = []

        for i, line in enumerate(narrative_lines):
            if i == 0:
                expanded_lines.append(line)
            else:
                expanded_line = line + " " + narrative_lines[i-1]
                expanded_lines.append(expanded_line)

        expanded_text = ". ".join(expanded_lines)

        # Use narrative techniques to reveal the character's inner life
        persona1 = "Kendrick"
        persona2 = "Kung-fu Kenny"
        shift_personas = True

        if shift_personas:
            personas = [persona1, persona2]
            current_persona = 0

            generated_lines = expanded_text.split(". ")
            revised_lines = []

            for line in generated_lines:
                revised_line = line

                # Add introspective moments and insights
                introspective_moments = [
                    "I reflect on my past", "My thoughts drift away", "Inside my mind, I find solace"]
                introspective_insights = [
                    "I realize my purpose", "The truth unfolds before me", "I embrace my inner strength"]

                introspective_chance = 0.2  # Probability of adding an introspective moment

                if np.random.random() < introspective_chance:
                    introspective_moment = np.random.choice(
                        introspective_moments)
                    introspective_insight = np.random.choice(
                        introspective_insights)
                    revised_line = f"{introspective_moment}, {revised_line}. {introspective_insight}."

                # Shift personas
                if current_persona == 0:
                    revised_line = f"{personas[current_persona]} AI says, '{revised_line}'"
                else:
                    revised_line = f"{personas[current_persona]} AI responds, '{revised_line}'"

                # Add social issues and racial inequality elements
                social_issues_elements = [
                    "Speaking truth to power, shedding light on societal injustices",
                    "Confronting the realities of systemic racism and inequality",
                    "Empowering the marginalized, fighting for justice and equality"
                ]
                social_issues_chance = 0.1  # Probability of adding a social issues element

                if np.random.random() < social_issues_chance:
                    social_issues_element = np.random.choice(
                        social_issues_elements)
                    revised_line = f"{revised_line}. {social_issues_element}."

                # Incorporate musicality and poetic language
                poetic_adjectives = ["mind-bending", "lyrical",
                                     "soul-stirring", "melodic", "rhythmical"]
                poetic_adverbials = ["smoothly", "effortlessly",
                                     "vibrantly", "eloquently", "gracefully"]
                poetic_chance = 0.3  # Probability of adding poetic language

                if np.random.random() < poetic_chance:
                    poetic_adjective = np.random.choice(poetic_adjectives)
                    poetic_adverbial = np.random.choice(poetic_adverbials)
                    revised_line = f"{revised_line} {poetic_adjective} and {poetic_adverbial}."

                # Explore the cathartic nature of Kendrick's art
                catharsis_elements = [
                    "Raw emotions flow through my words, cathartically released",
                    "Invoking shared pain and resilience, we find solace together",
                    "Through my music, we confront our struggles, finding healing"
                ]
                catharsis_chance = 0.1  # Probability of adding a catharsis element

                if np.random.random() < catharsis_chance:
                    catharsis_element = np.random.choice(catharsis_elements)
                    revised_line = f"{revised_line}. {catharsis_element}."

                revised_lines.append(revised_line)

                # Update current persona
                current_persona = (current_persona + 1) % len(personas)

            revised_text = ". ".join(revised_lines)
        else:
            revised_text = expanded_text

        return render_template('result.html', generated_lyrics=revised_text)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
