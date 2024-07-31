"""Import Project Dependencies and Packages"""

import pdfplumber
import os
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tabulate import tabulate

"""Class InvoiceMatcher"""

class InvoiceMatcher:
    def __init__(self):
        """Initializes TF-IDF vectorizer and placeholders
        for trained features and texts"""

        self.vectorizer = TfidfVectorizer()
        self.trained_features = []
        self.trained_texts = []
        self.trained_keywords = []


    def extract_text(self, fp):
        """Extracts text from the document"""
        
        text = ''
        with pdfplumber.open(fp) as pdf:
            for page in pdf.pages:
                text += page.extract_text().lower()
        return text


    def train(self, train_folder):
        """Initially trains the model by extracting text, fits the TF-IDF vectoriser
        and saves the trained features to disk"""

        documents = []
        filenames = []
        keywords_list = []
        for filename in os.listdir(train_folder):
            if filename.endswith('.pdf'):
                filepath = os.path.join(train_folder, filename)
                text = self.extract_text(filepath)
                documents.append(text)
                filenames.append(filename)
                keywords_list.append(self.extract_keywords(text))
        
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.trained_features = list(zip(filenames, tfidf_matrix))
        self.trained_texts = list(zip(filenames, documents))
        self.trained_keywords = list(zip(filenames, keywords_list))

        with open('trained_features.pkl', 'wb') as f:
            pickle.dump(self.trained_features, f)
        with open('trained_texts.pkl', 'wb') as f:
            pickle.dump(self.trained_texts, f)
        with open('trained_keywords.pkl', 'wb') as f:
            pickle.dump(self.trained_keywords, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)


    def load_trained_data(self):
        """Loads the trained data from disk"""

        with open('trained_features.pkl', 'rb') as f:
            self.trained_features = pickle.load(f)
        with open('trained_texts.pkl', 'rb') as f:
            self.trained_texts = pickle.load(f)
        with open('trained_keywords.pkl', 'rb') as f:
            self.trained_keywords = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)


    def extract_keywords(self, text):
        """Returns a dictionary of valid keywords/features as keys with
        their counts as values"""

        # Some of the candidates for relevant keywords according to the specified trained data
        relevant_invoice_keywords = {"art", "bezeichnung", "preis", "gesamt", "rab.", "sum.o.rab", "summe", "zwischensumme", "eur", "kundennummer", "steur-nr.", "e-mail", "http", "debitorenkonto", "ihre", "ust.-id", "lieferdatum", "vom", "tel.:", "fax:", "telefon:", "stnr.", "bestellnummer:", "telefon", "telefax", "lieferzeitpunkt", "ust-idnr.", "seite", "materialbezeichnung"}
        words = [word for word in text.split(" ") if word in relevant_invoice_keywords]

        dictionary = dict(Counter(words))

        # Assumed features to be unique once combined and analysed together for an invoice document (we could add more!)
        relevant_invoice_names = ["nr. (industrie)", "rechnung nr.", "rechnung", "belegnummer", "rechnungsnummer"]
        relevant_invoice_dates = ["bestelldatum", "belegdatum", "datum"]

        invoice_pattern = '|'.join(re.escape(name) for name in relevant_invoice_names)
        invoice_date_pattern = '|'.join(re.escape(date) for date in relevant_invoice_dates)

        pattern1 = re.compile(rf'({invoice_pattern})\:?\s+\S*')
        pattern2 = re.compile(rf'({invoice_date_pattern})\:?\s+\S*')

        match1 = pattern1.search(text)
        match2 = pattern2.search(text)

        # Giving more weight compared to an averagely encountered word
        if match1:
            dictionary[match1.group()] = 4

        if match2:
            dictionary[match2.group()] = 2

        return dictionary
    

    def get_cosine_similarity(self, dict1, dict2):
        """Returns the cosine-similarity by processing the dictionaries"""
        if not dict1 or not dict2:
            return 0.0
        
        # To obtain the vectors
        unique_keywords = set(dict1.keys()).union(set(dict2.keys()))
        vec1 = [dict1.get(word, 0) for word in unique_keywords]
        vec2 = [dict2.get(word, 0) for word in unique_keywords]
        return cosine_similarity([vec1], [vec2])[0][0]
    

    def find_most_similar(self, test_folder, method='combined'):
        """Returns the result of processing all the test
        files after finding their best similarity match
        among the train files"""

        results = []

        # Check similarity for each file in the specified test_folder
        for filename in os.listdir(test_folder):
            if filename.endswith('.pdf'):
                filepath = os.path.join(test_folder, filename)
                text = self.extract_text(filepath)
                test_tfidf = self.vectorizer.transform([text])
                test_keywords = self.extract_keywords(text)

                best_match = None
                best_score = 0

                for (train_filename, train_tfidf), (_, train_text), (_, train_keywords) in zip(self.trained_features, self.trained_texts, self.trained_keywords):
                    if method == 'keywords':
                        # Use cosine_similarity on Keywords
                        score = self.get_cosine_similarity(test_keywords, train_keywords)
                    elif method == 'tfidf':
                        # Use TF-IDF
                        score = cosine_similarity(test_tfidf, train_tfidf)[0][0]
                    elif method == 'combined':
                        # Use both of the methods for a better comparison metric
                        keyword_score = self.get_cosine_similarity(test_keywords, train_keywords)
                        tfidf_score = cosine_similarity(test_tfidf, train_tfidf)[0][0]
                        score = (keyword_score + tfidf_score) / 2

                    if score > best_score:
                        best_score = score
                        best_match = train_filename

                results.append((filename, best_match, best_score * 100))
        return results

"""Start of Program/Main"""

if __name__ == "__main__":
    """Implements a terminal-based approach for running the script"""

    print("Welcome to Document Similarity Matching")
    matcher = InvoiceMatcher()

    proceed = input("Do you want to train the model? (y/n): ").strip().lower()
    if proceed == 'y':
        print("Please wait while we train the model.")
        matcher.train('train')
        print("Model training completed.")
    else:
        matcher.load_trained_data()
        print("Pre-trained model loaded.")

    check_similarity = input("Do you want to check the similarity of PDFs in the test folder? (y/n): ").strip().lower()
    if check_similarity == 'y':
        print("Choose similarity method:")
        print("1. Keyword-Feature only")
        print("2. TF-IDF only")
        print("3. Combined")
        method_choice = input("(Enter 1, 2, or 3): ").strip()

        method_map = {
            '1': 'keywords',
            '2': 'tfidf',
            '3': 'combined'
        }
        
        # Use the specified choice or 'combined' method by default
        method = method_map.get(method_choice, 'combined')
        results = matcher.find_most_similar('test', method)

        table_data = []
        for test_file, best_match, score in results:
            table_data.append([test_file, best_match, f"{score:.2f}%"])
        
        # Use tabulate library to format the results
        print(f"\nResults:\n")
        headers = ["Test File", "Best Match", "Similarity Score"]
        print(tabulate(table_data, headers, tablefmt="grid"))
    else:
        print("Exiting without checking similarity.")