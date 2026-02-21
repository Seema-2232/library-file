import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Library:
    def __init__(self, filename="books.txt"):
        self.filename = filename
        self.books = []
        self.load_books()

    def load_books(self):
        if os.path.exists(self.filename):
            with open(self.filename, "r") as file:
                self.books = [line.strip() for line in file]

    def save_books(self):
        with open(self.filename, "w") as file:
            for book in self.books:
                file.write(book + "\n")

    def add_book(self, book_name):
        self.books.append(book_name)
        self.save_books()

    def print_books(self):
        for i, book in enumerate(self.books, 1):
            print(f"{i}. {book}")

    def get_number_of_books(self):
        return len(self.books)

    # ---------------- AI FEATURES ---------------- #

    def recommend_books(self, keyword, top_n=3):
        if not self.books:
            return []

        # Combine keyword with book list
        documents = self.books + [keyword]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        scores = similarity.flatten()
        ranked_indices = scores.argsort()[::-1]

        recommendations = []
        for i in ranked_indices[:top_n]:
            if scores[i] > 0:
                recommendations.append(self.books[i])

        return recommendations

    def suggest_related(self, book_name, top_n=3):
        if book_name not in self.books:
            return []

        documents = self.books
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity_matrix = cosine_similarity(tfidf_matrix)

        index = self.books.index(book_name)
        scores = similarity_matrix[index]

        ranked_indices = scores.argsort()[::-1]

        suggestions = []
        for i in ranked_indices[1:top_n+1]:
            if scores[i] > 0:
                suggestions.append(self.books[i])

        return suggestions