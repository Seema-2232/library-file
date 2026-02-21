import tkinter as tk
from tkinter import messagebox, ttk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SmartLibrary:

    def __init__(self, filename="books.txt"):
        self.filename = filename
        self.books = []
        self.load_books()

    def load_books(self):
        if os.path.exists(self.filename):
            with open(self.filename, "r") as file:
                self.books = [line.strip() for line in file]
        else:
            open(self.filename, "w").close()

    def save_books(self):
        with open(self.filename, "w") as file:
            for book in self.books:
                file.write(book + "\n")

    def add_book(self, book_name):
        self.books.append(book_name)
        self.save_books()

    def recommend(self, keyword, top_n=3):
        if not self.books:
            return []

        documents = self.books + [keyword]

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
        scores = similarity.flatten()
        ranked = scores.argsort()[::-1]

        results = []
        for i in ranked[:top_n]:
            if scores[i] > 0:
                results.append(self.books[i])
        return results

    def suggest_related(self, book_name, top_n=3):
        if book_name not in self.books:
            return []

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(self.books)
        similarity_matrix = cosine_similarity(tfidf)

        index = self.books.index(book_name)
        scores = similarity_matrix[index]
        ranked = scores.argsort()[::-1]

        results = []
        for i in ranked[1:top_n+1]:
            if scores[i] > 0:
                results.append(self.books[i])
        return results


class LibraryUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Library AI")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        self.library = SmartLibrary()
        self.create_widgets()
        self.refresh_list()

    def create_widgets(self):

        title = tk.Label(self.root, text="Smart Library Management System",
                         font=("Arial", 18, "bold"))
        title.pack(pady=10)

        # Add Book Section
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Label(frame, text="Book Name:", font=("Arial", 12)).grid(row=0, column=0, padx=5)

        self.book_entry = tk.Entry(frame, width=30, font=("Arial", 12))
        self.book_entry.grid(row=0, column=1, padx=5)

        tk.Button(frame, text="Add Book", command=self.add_book).grid(row=0, column=2, padx=5)

        # Book List
        self.listbox = tk.Listbox(self.root, width=80, height=10)
        self.listbox.pack(pady=10)

        # Recommendation Section
        rec_frame = tk.Frame(self.root)
        rec_frame.pack(pady=10)

        tk.Label(rec_frame, text="Keyword:", font=("Arial", 12)).grid(row=0, column=0, padx=5)

        self.keyword_entry = tk.Entry(rec_frame, width=20)
        self.keyword_entry.grid(row=0, column=1, padx=5)

        tk.Button(rec_frame, text="Recommend",
                  command=self.recommend_books).grid(row=0, column=2, padx=5)

        tk.Button(rec_frame, text="Suggest Related",
                  command=self.suggest_books).grid(row=0, column=3, padx=5)

        # Result Box
        self.result_box = tk.Text(self.root, height=6, width=80)
        self.result_box.pack(pady=10)

    def add_book(self):
        name = self.book_entry.get().strip()

        if name == "":
            messagebox.showwarning("Warning", "Book name cannot be empty")
            return

        self.library.add_book(name)
        self.book_entry.delete(0, tk.END)
        self.refresh_list()

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for book in self.library.books:
            self.listbox.insert(tk.END, book)

    def recommend_books(self):
        keyword = self.keyword_entry.get().strip()
        results = self.library.recommend(keyword)

        self.result_box.delete("1.0", tk.END)
        if results:
            self.result_box.insert(tk.END, "Recommended Books:\n")
            for book in results:
                self.result_box.insert(tk.END, f"- {book}\n")
        else:
            self.result_box.insert(tk.END, "No matching books found.")

    def suggest_books(self):
        selected = self.listbox.curselection()
        self.result_box.delete("1.0", tk.END)

        if not selected:
            self.result_box.insert(tk.END, "Select a book from the list first.")
            return

        book_name = self.listbox.get(selected[0])
        results = self.library.suggest_related(book_name)

        if results:
            self.result_box.insert(tk.END, "Related Books:\n")
            for book in results:
                self.result_box.insert(tk.END, f"- {book}\n")
        else:
            self.result_box.insert(tk.END, "No related books found.")


if __name__ == "__main__":
    root = tk.Tk()
    app = LibraryUI(root)
    root.mainloop()