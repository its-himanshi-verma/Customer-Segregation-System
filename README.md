# 🛍️ Customer Segmentation System Using Machine Learning

## 📌 Project Overview

The **Customer Segmentation System** is a machine learning project that uses unsupervised learning techniques to group customers based on their demographic characteristics and spending behavior.

Businesses can use customer segmentation to better understand their customers, identify spending patterns, and develop targeted marketing strategies.

This project uses the **Mall Customer Segmentation Dataset** and applies clustering algorithms to discover meaningful customer groups.

## 🎯 Objectives

- Preprocess and clean customer data.
- Explore customer demographics and spending patterns.
- Scale numerical features for effective clustering.
- Apply clustering algorithms to segment customers.
- Determine the optimal number of clusters using evaluation techniques.
- Visualize and interpret customer segments.
- Generate actionable business insights from the results.

## 📊 Dataset

**Dataset:** Mall Customer Segmentation Data

**Source:** [Kaggle – Customer Segmentation Tutorial in Python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

The dataset contains customer information collected from a shopping mall.

### Features

| Feature | Description |
|---|---|
| CustomerID | Unique identifier for each customer |
| Gender | Gender of the customer |
| Age | Age of the customer |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1–100) | Spending score assigned by the mall |

**Note:** CustomerID is an identifier and should generally be excluded from clustering. The selected features depend on the segmentation objective.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook / Google Colab

## 🧠 Machine Learning Techniques

This project focuses on **unsupervised learning**, where the algorithm discovers patterns in data without predefined target labels.

### 1. Data Preprocessing
- Load the dataset and inspect its structure.
- Check for missing values and duplicate records.
- Explore feature distributions and relationships.
- Select relevant features for clustering.
- Encode categorical features when required.

### 2. Feature Scaling
Scale numerical features using techniques such as `StandardScaler` or `MinMaxScaler`.

Scaling prevents features with larger numerical ranges from dominating distance-based clustering algorithms.

### 3. Clustering Algorithms

**K-Means Clustering**

Groups customers into a predefined number of clusters by minimizing the within-cluster sum of squared distances.

**Hierarchical Clustering (Optional)**

Builds a hierarchy of customer groups and can help explore different levels of segmentation.

**DBSCAN (Optional)**

Identifies dense groups of customers and can detect outliers without requiring a predefined number of clusters.

### 4. Finding the Optimal Number of Clusters

The project can use the following evaluation methods:

- **Elbow Method:** Identifies a potential cluster count by examining changes in within-cluster sum of squares (WCSS).
- **Silhouette Score:** Measures how well customers fit within their assigned clusters compared with other clusters.

The final number of clusters should be selected based on evaluation results and the interpretability of the resulting customer groups.

### 5. Customer Segment Interpretation

Visualize the clusters and compare their characteristics using age, annual income, and spending score.

Possible segment types include:

- High-income, high-spending customers
- High-income, low-spending customers
- Low-income, high-spending customers
- Low-income, low-spending customers
- Average-income, average-spending customers

These are illustrative segment descriptions. The actual segments must be determined from the model's results.

## 📁 Project Structure

```text
Customer-Segmentation-System/
│
├── data/
│   └── Mall_Customers.csv
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
├── visualizations/
│   ├── elbow_method.png
│   └── customer_clusters.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

*The structure above is a suggested organization. Adjust filenames and folders to match your actual repository.*

## ⚙️ Installation and Setup

### Prerequisites

Install Python 3.9 or later and Git.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Customer-Segmentation-System.git
cd Customer-Segmentation-System
```

Replace `YOUR-USERNAME` with your GitHub username and update the repository name if necessary.

### Step 2: Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

### Step 4: Download the Dataset

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).
2. Download the dataset.
3. Extract `Mall_Customers.csv`.
4. Place it in the `data/` directory.

### Step 5: Run the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `notebooks/customer_segmentation.ipynb` and execute the cells in order.

## 📈 Expected Visualizations

The project can include the following visualizations:

1. **Exploratory Data Analysis:** Distributions of age, annual income, and spending score.
2. **Correlation Heatmap:** Relationships between numerical features.
3. **Elbow Curve:** WCSS against the number of clusters.
4. **Silhouette Analysis:** Comparison of clustering quality for candidate cluster counts.
5. **Customer Cluster Plot:** A scatter plot showing customer groups by annual income and spending score.
6. **Cluster Profiles:** Summary statistics describing each customer segment.

## 💡 Business Applications

Customer segmentation can help businesses:

- Design personalized marketing campaigns.
- Identify high-value customers.
- Develop customer retention strategies.
- Improve promotional offers and loyalty programs.
- Understand differences in customer spending behavior.
- Allocate marketing resources more effectively.

## ✅ Expected Outcomes

After completing this project, the following outcomes are expected:

- A processed and scaled customer dataset.
- One or more clustering models.
- An evidence-based selection of the number of clusters.
- Visualizations of the identified customer groups.
- Interpretable customer segment profiles.
- Business recommendations based on observed customer behavior.

*Actual cluster counts, evaluation scores, and business findings should be added after running the analysis.*

## 🔮 Future Improvements

- Compare multiple clustering algorithms systematically.
- Use additional customer purchase-history data.
- Build an interactive dashboard using Streamlit.
- Develop a customer segmentation interface for business users.
- Evaluate cluster stability on additional datasets.
- Create personalized marketing recommendations for each segment.

## 👩‍💻 Learning Outcomes

This project demonstrates practical knowledge of:

- Data preprocessing and exploratory data analysis.
- Feature engineering and feature scaling.
- Unsupervised machine learning.
- K-Means and other clustering algorithms.
- Cluster evaluation and model selection.
- Data visualization and business interpretation.

## 📚 References

- [Kaggle – Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)

## 📄 License

This project is intended for educational and learning purposes. If you distribute the dataset or reuse its contents, follow the dataset's applicable license and usage terms.

---

**⭐ If you find this project useful, consider giving the repository a star!**
