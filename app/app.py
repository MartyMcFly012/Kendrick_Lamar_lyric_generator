from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)
app._static_folder = 'static'

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

# Assuming transition_matrix is a NumPy array
# Replace NaN values with 0
transition_matrix[np.isnan(transition_matrix)] = 0

# Normalize transition matrix
transition_matrix = transition_matrix / \
    transition_matrix.sum(axis=1, keepdims=True)


def sample_next_word(seed_word_id, diversity=.7, top_p=0.8):
    next_word_probs = transition_matrix[seed_word_id]
    sorted_word_ids = np.argsort(next_word_probs)[::-1]
    cumulative_probs = np.cumsum(next_word_probs[sorted_word_ids])
    sorted_indices = np.arange(len(next_word_probs))[::-1]

    # Truncate to keep only the most likely words whose cumulative probability exceeds top_p
    sorted_indices = sorted_indices[cumulative_probs > top_p]

    # Apply diversity to the remaining words
    scaled_probs = next_word_probs[sorted_word_ids[sorted_indices]]
    scaled_probs = np.power(scaled_probs, diversity)
    scaled_probs /= np.sum(scaled_probs)

    # Select a word randomly based on the scaled probabilities
    next_word_id = np.random.choice(
        sorted_word_ids[sorted_indices], p=scaled_probs)
    return next_word_id


def beam_search(seed_sequence, num_words, diversity=.275, beam_width=3):
    sequences = [(seed_sequence, 0.0)]

    for _ in range(num_words):
        candidates = []

        for sequence, score in sequences:
            next_word_id = sample_next_word(
                sequence[-1], diversity=diversity)
            candidate_sequence = sequence + [next_word_id]
            candidate_score = score + \
                np.log(transition_matrix[sequence[-1]][next_word_id])

            candidates.append((candidate_sequence, candidate_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        sequences = candidates[:beam_width]

    generated_sequence = sequences[0][0]
    return generated_sequence


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        seed_text = request.form['seed_text']
        num_words = int(request.form['num_words'])
        diversity = float(request.form['diversity'])

        # Step 6: Text Generation
        seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        generated_sequence = beam_search(
            seed_sequence, num_words, beam_width=3, diversity=.275)

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

        # Reflect Kendrick's lyrical style and themes
        kendrick_themes = [
            "struggle", "hope", "justice", "racism", "equality", "perseverance", "consciousness", "truth", "resilience", "unity"
        ]

        for theme in kendrick_themes:
            if theme in revised_text.lower():
                revised_text = revised_text.replace(
                    theme, f"'{theme.capitalize()}'")

        # Adjust lyrical style for authenticity
        revised_text = revised_text.replace(" i ", " I ")
        revised_text = revised_text.replace(" ai ", " I ")
        revised_text = revised_text.replace("i'm", "I'm")
        revised_text = revised_text.replace(" im ", " I'm ")
        revised_text = revised_text.replace("i've", "I've")
        revised_text = revised_text.replace("i'll", "I'll")
        revised_text = revised_text.replace("i'd", "I'd")
        revised_text = revised_text.replace("i'd", "I'd")
        revised_text = revised_text.replace(" imma ", " I'ma ")
        revised_text = revised_text.replace("gonna", "gon'")
        # Add sentence breaks (double line breaks) to improve text flow
        revised_text = revised_text.replace(". ", ".\n\n")

        # Render the revised lyrics
        return render_template('result.html', generated_lyrics=revised_text)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
