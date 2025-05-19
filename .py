import pandas as pd
import numpy as np
from tkinter import Tk, filedialog, Button, Label, ttk, StringVar, Entry, messagebox, Scrollbar
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class EcommerceModelApp:
    def __init__(self, master):
        self.master = master
        master.title("E-commerce Rating Predictor")

        self.label = Label(master, text="Upload E-commerce CSV file")
        self.label.pack()

        self.upload_button = Button(master, text="Upload CSV", command=self.load_csv)
        self.upload_button.pack()

        self.add_more_button = Button(master, text="Add More CSV", command=self.add_csv)
        self.add_more_button.pack()

        self.sort_label = Label(master, text="Sort by column (e.g., rating or title):")
        self.sort_label.pack()
        self.sort_column = StringVar()
        self.sort_entry = Entry(master, textvariable=self.sort_column)
        self.sort_entry.pack()

        self.sort_button = Button(master, text="Sort Data", command=self.sort_data)
        self.sort_button.pack()

        self.train_button = Button(master, text="Train Model", command=self.train_model)
        self.train_button.pack()

        self.result_label = Label(master, text="")
        self.result_label.pack()

        # Treeview and Scrollbars
        self.tree_frame = ttk.Frame(master)
        self.tree_frame.pack(fill='both', expand=True)

        self.tree = ttk.Treeview(self.tree_frame)
        self.tree.pack(side='left', fill='both', expand=True)

        self.v_scroll = Scrollbar(self.tree_frame, orient='vertical', command=self.tree.yview)
        self.v_scroll.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=self.v_scroll.set)

        self.h_scroll = Scrollbar(master, orient='horizontal', command=self.tree.xview)
        self.h_scroll.pack(side='bottom', fill='x')
        self.tree.configure(xscrollcommand=self.h_scroll.set)

        self.df = pd.DataFrame()

    def preprocess_csv(self, df):
        numeric_cols = ['price', 'num_reviews', 'rating']
        text_cols = ['title', 'description', 'category']

        if "rating" not in df.columns:
            np.random.seed(42)
            df["rating"] = np.round(np.random.uniform(1, 5, size=len(df)), 1)

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)

        for col in text_cols:
            if col in df.columns:
                df[col].fillna('', inplace=True)

        return df

    def load_csv(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                new_df = pd.read_csv(file_path)
                new_df = self.preprocess_csv(new_df)
                self.df = new_df
                self.display_data(self.df.head())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

    def add_csv(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                new_df = pd.read_csv(file_path)
                new_df = self.preprocess_csv(new_df)
                self.df = pd.concat([self.df, new_df], ignore_index=True)
                self.display_data(self.df.head())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add CSV:\n{e}")

    def display_data(self, df):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"

        for col in df.columns:
            self.tree.heading(col, text=col, anchor='w')
            self.tree.column(col, width=120, anchor='w')

        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def sort_data(self):
        if self.df.empty:
            messagebox.showerror("Error", "No data to sort. Upload a CSV first.")
            return

        column = self.sort_column.get().strip()
        if column not in self.df.columns:
            messagebox.showerror("Error", f"Column '{column}' not found in data.")
            return

        try:
            self.df = self.df.sort_values(by=column, ascending=True)
            self.display_data(self.df.head(50))
        except Exception as e:
            messagebox.showerror("Error", f"Could not sort data:\n{e}")

    def train_model(self):
        if self.df.empty:
            messagebox.showerror("Error", "No data loaded.")
            return

        required_features = ["title", "description", "price", "category", "num_reviews"]
        target = "rating"

        missing_cols = [col for col in required_features if col not in self.df.columns]
        if missing_cols:
            messagebox.showerror("Error", f"Missing required columns: {', '.join(missing_cols)}")
            return

        try:
            X = self.df[required_features]
            y = self.df[target]

            preprocessor = ColumnTransformer(transformers=[
                ("title", TfidfVectorizer(max_features=500), "title"),
                ("desc", TfidfVectorizer(max_features=500), "description"),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
                ("num", StandardScaler(), ["price", "num_reviews"])
            ])

            model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            self.result_label.config(text=f"Model Trained\nRMSE: {rmse:.2f}, R²: {r2:.2f}")
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed:\n{str(e)}")

if __name__ == "__main__":
    root = Tk()
    root.geometry("1000x600")
    app = EcommerceModelApp(root)
    root.mainloop()
