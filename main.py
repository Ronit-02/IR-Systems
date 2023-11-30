## Point System

## Soundex - 1
# same = 0
# diff = 1

# pints = 0,1


## Edit Distance - 5
# len = 10
# dist = 3
# val = 10-3 = 7 (70% of 10)

# points = (len - ed)/len * 5 = 2


## Cosine Similarity - 10
# 1 -> 10
# 0.12 -> 10 * 1.2

# points = cs * 10


## We will rate text similarity based on above point system (0, 16)

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download stopwords and punkt for cosine similarity
nltk.download('stopwords')
nltk.download('punkt')

def soundex(query):

    # Step 0: Clean up the query string
    query = query.lower()
    letters = [char for char in query if char.isalpha()]

    # Step 1: Save the first letter. Remove all occurrences of a, e, i, o, u, y, h, w.
    # If query contains only 1 letter, return query+"000"
    if len(query) == 1:
        return query + "000"

    to_remove = ('a', 'e', 'i', 'o', 'u', 'y', 'h', 'w')

    first_letter = letters[0]
    letters = letters[1:]
    letters = [char for char in letters if char not in to_remove]

    if len(letters) == 0:
        return first_letter + "000"

    # Step 2: Replace all consonants (include the first letter) with digits according to rules
    to_replace = {('b', 'f', 'p', 'v'): 1, ('c', 'g', 'j', 'k', 'q', 's', 'x', 'z'): 2,
                  ('d', 't'): 3, ('l',): 4, ('m', 'n'): 5, ('r',): 6}

    first_letter = [value if first_letter else first_letter for group, value in to_replace.items()
                    if first_letter in group]
    letters = [value if char else char
               for char in letters
               for group, value in to_replace.items()
               if char in group]

    # Step 3: Replace all adjacent same digits with one digit.
    letters = [char for ind, char in enumerate(letters)
               if (ind == len(letters) - 1 or (ind+1 < len(letters) and char != letters[ind+1]))]

    # Step 4: If the saved letter’s digit is the same the resulting first digit, remove the digit (keep the letter)
    if first_letter == letters[0]:
        letters[0] = query[0]
    else:
        letters.insert(0, query[0])

    # Step 5: Append 3 zeros if result contains less than 3 digits.
    # Remove all except first letter and 3 digits after it.
    first_letter = letters[0]
    letters = letters[1:]

    letters = [char for char in letters if isinstance(char, int)][0:3]

    while len(letters) < 3:
        letters.append(0)

    letters.insert(0, first_letter)
    string = "".join([str(l) for l in letters])
    return string


def edit_distance(word1, word2):

    word2 = word2.lower()
    word1 = word1.lower()
    matrix = [[0 for x in range(len(word2) + 1)] for x in range(len(word1) + 1)]

    for x in range(len(word1) + 1):
        matrix[x][0] = x
    for y in range(len(word2) + 1):
        matrix[0][y] = y

    for x in range(1, len(word1) + 1):
        for y in range(1, len(word2) + 1):
            if word1[x - 1] == word2[y - 1]:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1],
                    matrix[x][y - 1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1] + 1,
                    matrix[x][y - 1] + 1
                )

    return matrix[len(word1)][len(word2)]


def cosine_similarity(text1, text2):

    # vector1 = np.array([text1])
    # vector2 = np.array([text2])

    # dot_product = np.dot(vector1, vector2)

    # magnitude1 = np.linalg.norm(vector1)
    # magnitude2 = np.linalg.norm(vector2)

    # cosine_similarity = dot_product / (magnitude1 * magnitude2)
    # return cosine_similarity

    stop_words = set(stopwords.words('english'))

    # Tokenize the text
    word_tokens = word_tokenize(text1 + " " + text2)

    # Filter out stop words
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

    # Create a set of unique words from the combined text
    unique_words = list(set(filtered_sentence))

    # Create vectors to represent word frequency in each text
    vector1 = [0] * len(unique_words)
    vector2 = [0] * len(unique_words)

    # Calculate word frequency for text 1
    for word in word_tokenize(text1):
        if word.lower() not in stop_words:
            vector1[unique_words.index(word.lower())] += 1

    # Calculate word frequency for text 2
    for word in word_tokenize(text2):
        if word.lower() not in stop_words:
            vector2[unique_words.index(word.lower())] += 1

    # Calculate cosine similarity manually
    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)

    cosine_similarity = dot_product / (magnitude_vector1 * magnitude_vector2)
    return cosine_similarity


# Streamlit app
def main():
    st.title('Text Rating App')

    text1 = st.text_area('Enter text phrase 1')
    text2 = st.text_area('Enter text phrase 2')

    if st.button('Calculate Ratings'):
        if text1 and text2:

            l = max(len(text1), len(text2))

            x1 = soundex(text1)
            x2 = soundex(text2)
            y = edit_distance(text1, text2)
            z = cosine_similarity(text1, text2)

            max_score = 16
            curr_score = (1 if x1==x2 else 0) + (l - y)/l * 5 + z * 10
            points = round(curr_score / max_score * 100)

            # Display the ratings
            st.subheader('Rating:')
            st.slider('', 0, 100, points)


        else:
            st.warning('Please enter both text phrases.')

# Streamlit main function
if __name__ == "__main__":
    main()