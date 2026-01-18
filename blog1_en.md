## Overview of Machine Learning
![Machine_Learning_broad](https://mindlabinc.ca/wp-content/uploads/2024/05/Machine-Learning.webp)
Before Machine Learning became widely popular, did you used to think sentient robots only appeared on Hollywood screens or in cartoons? 

Now, the "*twist*" of Machine Learning is present in our daily lives. This is the era of Machine Learning, where simply scrolling through your phone, entering a search, saving, or liking content are all recorded as data by the machine.

> "Machine learning is not magic, and it’s not menace. It’s mathematics with purpose"
> -- ScienceNewsToday

# 1. Overview of Machine Learning
The origin of Machine Learning began in the 1940s-1950s when Alan Turing posed the question: "Can machines think like humans?" which began to permeate the scientific community. 

In 1943, McCulloch and Pitts proposed the first artificial neuron model. This was the most significant turning point for computers and laid the foundation for Machine Learning.

## 1.1. What is Machine Learning?
**Machine Learning is a computer system that learns from data points through a training process**. Once the machine has learned and improved from patterns, it creates algorithmic models that can infer from new data, thereby proposing predictions with high accuracy. 

For example, Machine Learning analyzes web surfing habits through frequently viewed content. It then suggests new videos or products with similar content.

## 1.2. Traditional Programming vs Machine Learning: Key Differences Explained
Systems in traditional programming operate based on simple, pre-written conditional statements such as if-else, case A-B, etc. Thus, the essence of traditional programming is that the machine only does exactly what is written, following stable rules for management and calculation. To change programming in the old way, one must "*fix*" the code. 

Compared to traditional programming, Machine Learning does not require explicit algorithmic descriptions. The core of building a Machine Learning model lies in the data. Machine Learning allows the system to utilize *INPUT* and predict *OUTPUT*, adapting and improving over time as new data is added, with accuracy based on probability.

## 1.3. A Comparison of AI, Machine Learning, and Deep Learning
|   | Artificial Intelligence | Deep Learning | Machine Learning |
|---|-----|-----|-----|
| Scope | The broadest field  | A part of AI | A specialized branch of Machine Learning |
| Goal  | Any system that can perform tasks requiring human intelligence | Learning from data and improving over time | dentifying complicated patterns |
| Data |   | Processing structured data | Processing more complicated unstructured data like images/videos (multi-layered neural networks) |
## 1.4. Why Is Machine Learning Important Today?
Scientists began creating programs for computers to analyze large volumes of data, adapt to errors from a constantly changing environment, and solve complex problems beyond human capability. 

Major current technologies all apply Machine Learning, such as climate prediction models, medical diagnosis, real-time autonomous transportation, and more.

## 1.5. An Overview of the Role of Machine Learning
**1. Decision making**
    - Classifying non-linear data, complex relationships
    - Identifying data patterns
A factor for predictive capabilities across various fields.

**2. Trend and Outcome Prediction**
    - Personalization and customization through experience, adjusting services, and recommendations
    - Recommendation systems, suggesting similar content

**3. Wide-Ranging Applications**
    - Education: Detecting signs of anomalies
    - Banking: Detecting abnormal transaction
    - Applications: Spotify suggests songs based on listening habits
    - Business: Predicting product demand, optimizing production processes 
    - Healthcare: Personal medicine assistant, analyzing X-ray and CT scans


## 2. Type of machine learning
![types_of_ML](/static/uploads/20260118_204736_c834b889.png)
Nowadays there are various types of machine learning algorithm to solve specific problems. Therefore, this part will have a quick explanation about two focused group are supervised learning, unsupervised learning and common algorithms are semi-supervised learning and reinforcement learning.

### 2.1. Supervised learning

Supervised learning is a task that uses labeled data to train the model to give this model have a ability to predict a relation between input and output.

![supervised](/static/uploads/20260118_204909_2b8164f8.jpg)

The example has pictures divided in two:
- One picture has **apple** label
- One picture has **strawberry** label
The model after training will predict new pictures (not included in training) is **apple** or **strawberry**

## 2.2. Unsupervised learning

Unsupervised learning has trained data only contains input data without corresponding outputs (unlabeled data).

This algorithm extracts important informations among the data and separate into different groups.

![Unsupervised](/static/uploads/20260118_204931_2fe2b3b0.webp)

Example has unlabeled pictures and model based on that to find groups have related informations
- Two apple pictures have relative shape
- Lemon and orange pictures have a whole fruit and a half one
- The picture has a whole watermelon and a slice of a watermelon into a different group

## 2.3. Semi-supervised learning

In reality labeled data is rare and difficult to do (high cost and plenty of time), and unlabeled data is numerous, where semi-supervised learning will shine.

Semi-supervised learning is combination between supervised learning and unsupervised learning because this algorithm includes labeled and unlabeled data.

The ultimate goal of this model algorithm is to predict output better than supervised learning and unsupervised learning which only use one type of data (labeled or unlabeled).

Example about news classification on newspaper websites or social networks:
- There are 200 articles labeled with topics (sports, economics, entertainment, ...)
- There are 100,000 unlabeled articles
- Learn language structure and topics from both data sources

## 2.4. Reinforcement learning

Reinforcement learning is a model agorithm allows automatic agent continuously take actions by interacting with the environment to optimize behavior. In other words learn by trial and error and take respond from enviroment.

Example about self-driving cars:
- State: Car position, speed, lane, surrounding vehicles
- Action: Turn left/right, accelerate/decelerate, brake
- Reward: +10 for safe movement, -100 for collision, -5 for unnecessary lane change
- The car learns to drive safely through millions of simulations
## 3. Machine Learning Algorithms

Each learning method is like a different "educational philosophy" for computers. If we view Machine Learning as a journey to solve a problem, then the learning methods are the "strategies," while the algorithms are the specific "weapons". Depending on the data characteristics and forecasting goals, we choose the most suitable weapon.

### 3.1. Regression Group – Predicting Numerical Values

**Goal:** Find a rule to predict a continuous variable.

* **Linear Regression:** It is not simply about finding a straight line; it also helps us understand the impact level of each input variable. A classic example is house price prediction; specifically, we can know exactly how many millions the total house value increases for every additional square meter of area.
* **Ridge & Lasso Regression:** These are "upgraded" versions of linear regression. They add mathematical components to prevent **Overfitting** (where the model performs well on old data but poorly on new data), making the model more robust.
* **Random Forest Regression:** Instead of using a single mathematical function, this algorithm combines results from hundreds of different "decision trees" to produce a final number, which is effective for data with many complex variables.

### 3.2. Classification Group – Identifying Objects

**Goal:** Determine which group data belongs to among pre-defined groups.

* **Logistic Regression:** Contrary to its name, this is the "king" of binary classification (0 and 1). It is very popular in medicine for predicting whether a patient has a disease based on test indicators.
* **Support Vector Machine (SVM):** This algorithm doesn't just divide boundaries but tries to create the widest possible "margin" between two groups. To visualize this, imagine building the widest possible moat to separate two opposing armies.
* **Naïve Bayes:** Based on Bayes' probability theorem, this algorithm is extremely fast and effective for text classification problems, such as filtering spam emails or analyzing customer sentiment via comments.
* **Neural Networks:** Inspired by biological neural networks in the human brain, this algorithm can learn complex rules (face/voice recognition, brainwave analysis). This is the foundation of Deep Learning – the tool behind current AI chatbots.

### 3.3. Clustering Group – Discovering Hidden Structures

**Goal:** Automatically group data based on similarity without labels.

* **K-Means:** The algorithm works by selecting "k" center points and "pulling" surrounding data points toward them. Through many iterations, clusters will gradually form clearly.
* **Gaussian Mixture Models (GMM):** Unlike K-Means (which clusters in circles), GMM is more flexible by treating each cluster as an oval distribution, allowing clusters to overlap more naturally.
* **Principal Component Analysis (PCA):** Although often used for dimensionality reduction, PCA helps us see the most "core" components of the data, thereby supporting more accurate and intuitive clustering on graphs.

## 4. The Steps of a Machine Learning Project
### 4.1. Problem Definition
This is the foundational step that determines success or failure, helping to avoid solving the wrong problem. Three factors need to be clarified:

* **Goal:** Determine if the problem type is Prediction (Regression), Classification, or Grouping (Clustering).
* **Practical Value:** Identify what pain point the result solves for the user or business.
* **Success Metrics:** Choose the right evaluation metrics from the start for objective acceptance.

### 4.2. Data Collection and Processing

This is the most time-consuming stage (60-80%), following the "Garbage In, Garbage Out" principle.

**4.2.1. Data Collection**

* **Sources:** Internal (SQL, Log), Public (Kaggle), Web Scraping (Code), API, or IoT/Sensors.
* **Formats:** Structured (Excel, SQL), Semi-structured (JSON, XML), Unstructured (Text, Images, Video).

**4.2.2. Data Pre-processing**

1. **Cleaning:** Handle missing data (impute/delete), remove duplicates, and fix formatting errors to ensure consistency.
2. **Integration:** Combine data from multiple sources, map schemas, and deduplicate across sources.
3. **Transformation:** Normalization (Scaling) for distance-based algorithms, Encoding categorical variables (One-hot/Label encoding), and Feature engineering.
4. **Reduction:** Dimensionality reduction (PCA), select important features, or sample representatives to speed up training.

### 4.3. Model Selection and Training

The goal is to find a balanced model, avoiding ones that are too simple (inaccurate) or too complex (overfitting).

* **Understand the problem:** Determine if the data is numerical or categorical, and whether it is labeled or unlabeled.
* **Model Suggestions:**
* *Regression:* Linear Regression, Decision Trees, Random Forest, Neural Networks.
* *Classification:* Logistic Regression, SVM, k-NN, Neural Networks.
* *Clustering:* k-Means, Hierarchical Clustering, DBSCAN.



### 4.4. Model Evaluation

Split the data into a **Training Set** to learn and a **Test Set** to evaluate on new data. **Cross-validation (k-fold)** should be used for a more objective evaluation.

**Evaluation Metrics:**

* *Regression:* MSE, MAE, R-squared.
* *Classification:* Accuracy, Precision, Recall, F1-score.

### 4.5. Model Selection Optimization Techniques

* **Grid Search:** Tests all parameter combinations. Accurate but resource-intensive.
* **Random Search:** Tests a random subset. Faster and comparable efficiency to Grid Search.
* **Bayesian Optimization:** Uses probability to predict and select the best parameters intelligently.
* **Based on Cross-validation:** Selects the model with the best average performance across multiple splits to avoid overfitting.

## 5. Building a Basic Machine Learning Model

We'll use Python and scikit-learn, which is super user-friendly for beginners and handles most of the heavy lifting without needing deep math knowledge right away. The goal here is to make this hands-on so you can try it yourself and see how the concepts from earlier sections come together in code.

### 5.1 Tools You’ll Need

Before we dive in, let's get set up. These are the basics you'll need – nothing fancy, and it's all free or easy to install.

- **Python**: Version 3.8 or higher. If you don't have it, download from python.org. It's the go-to language for ML because of its simplicity and huge community support.
- **Environment**: I recommend Jupyter Notebook (install via Anaconda for ease) because it lets you run code in cells, see outputs immediately, and mix in explanations. Alternatively, Google Colab is awesome – it's online, free, and you don't install anything; just sign in with Google and start coding. It's great for sharing notebooks too.
- **Libraries**: These are Python packages that add ML superpowers. Install them once in your terminal or Colab with:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn
  ```
  - **numpy**: For handling arrays and math operations efficiently (think of it as Excel on steroids for numbers).
  - **pandas**: Makes data easy to work with, like loading, cleaning, and exploring tables.
  - **matplotlib and seaborn**: For plotting graphs and visuals – seaborn is basically a prettier version of matplotlib.
  - **scikit-learn**: The star of the show. It has built-in datasets, algorithms, and tools for training/evaluating models. No need for neural networks yet; this is for classic ML.

If you're new to installing, search "install Jupyter Notebook" or just use Colab to skip that hassle.

### 5.2 The Example Problem – Iris Flower Classification

For our example, we're tackling the Iris dataset, which is like the "Hello World" of machine learning. It was collected by biologist Ronald Fisher in 1936 and has been used in countless tutorials since.

- **Why this dataset?** It's small (only 150 samples), clean (no missing values or mess to fix), and balanced (50 samples per species). This lets us focus on learning the process without getting stuck on data cleaning. In real life, datasets are messier, but starting simple builds confidence.
- **Features (inputs)**: Four measurements from the flower:
  - Sepal length (the long outer leaves, in cm)
  - Sepal width
  - Petal length (the colorful inner parts)
  - Petal width
  These are numerical, which is perfect for ML algorithms.
- **Target (output)**: The species of the flower – one of three classes:
  - Setosa (easy to spot, usually separate in plots)
  - Versicolor
  - Virginica (these two overlap more, making it a bit challenging)
- **Task type**: This is supervised classification – we have labeled data (we know the species for each sample), and the model learns to map features to labels.
- **Real-world connection**: Think of it like identifying plant types from photos or measurements in agriculture, or similar tasks in medicine (e.g., classifying tumors from scans).

The dataset is baked into scikit-learn, so we can load it with one line. If you want to explore it outside code, you can find it on Kaggle or UCI ML Repository.

### 5.3 Step-by-Step Code

Now, the fun part: the code. I'll show the full script with comments explaining what's happening. It follows the 5-step workflow exactly. Copy this into a Jupyter/Colab notebook and run it cell by cell – you'll see data tables, plots, and results pop up.

I've chosen K-Nearest Neighbors (KNN) as our algorithm because it's intuitive: it classifies a new point by looking at its "k" closest neighbors in the data and taking a majority vote. No black-box magic; you can visualize why it works.

```python
# 1. Import everything we need – these are like toolboxes for data and ML
import numpy as np  # For numerical arrays and math
import pandas as pd  # For dataframes (like spreadsheets)
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns  # Fancier plots

from sklearn.datasets import load_iris  # Built-in dataset
from sklearn.model_selection import train_test_split  # To split data
from sklearn.preprocessing import StandardScaler  # To normalize features
from sklearn.neighbors import KNeighborsClassifier  # Our ML algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Evaluation tools

# 2. Load and explore the data (Step 2: Data Collection & Preprocessing)
iris = load_iris()  # Loads the dataset as a bunch object
X = iris.data       # Features: 150 rows x 4 columns (measurements)
y = iris.target     # Labels: 150 values (0=Setosa, 1=Versicolor, 2=Virginica)

# Convert to a pandas DataFrame for easier viewing and analysis
df = pd.DataFrame(X, columns=iris.feature_names)  # Add column names
df['species'] = pd.Categorical.from_codes(y, iris.target_names)  # Add human-readable species

# Quick inspection: first few rows and summary
print("First 5 rows of data:")
print(df.head())
print("\nData summary:")
print(df.describe())  # Stats like mean, min, max
print("\nSpecies counts:")
print(df['species'].value_counts())  # Should be 50 each – balanced!

# Visualize to understand relationships (e.g., petals separate species well)
sns.pairplot(df, hue='species', palette='husl')  # Scatter plots for all pairs of features, colored by species
plt.suptitle("Iris Data Pairplot")
plt.show()

# 3. Prepare the data: split and scale (still Step 2)
# Split: 80% for training (learning), 20% for testing (evaluation) – random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratify keeps species balanced in splits
)

# Scale features: KNN uses distances, so we normalize to mean=0, std=1 to avoid big features dominating
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train, transform train
X_test = scaler.transform(X_test)        # Transform test (don't fit on test to avoid data leakage)

# 4. Train the model (Step 3: Model Selection & Training)
# KNN: classifies based on k nearest points' votes
model = KNeighborsClassifier(n_neighbors=5)  # Start with k=5 – a common default
model.fit(X_train, y_train)  # This is where the "learning" happens – it memorizes the training data

# 5. Evaluate (Step 4: Model Evaluation)
y_pred = model.predict(X_test)  # Get predictions on test set

# Basic metric: accuracy (fraction correct)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc:.3f} (that's {acc*100:.1f}%)")

# Detailed report: precision (avoid false positives), recall (avoid false negatives), f1 (balance)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix: shows where mistakes happen (e.g., Versicolor misclassified as Virginica)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix – How Well Did We Do?')
plt.show()

# 6. Improve (Step 5: Improvement & Deployment)
# Tune k: try values 1-14, see which gives best accuracy
accuracies = []
for k in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)  # Built-in score method for accuracy
    accuracies.append(acc)

# Plot to visualize – peak is the best k
plt.plot(range(1,15), accuracies, marker='o', linestyle='--')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Test Accuracy')
plt.title('Tuning k for Better Performance')
plt.grid(True)
plt.show()

best_k = np.argmax(accuracies) + 1  # +1 because range starts at 1
print(f"Best k = {best_k} with accuracy = {max(accuracies):.3f}")

# For deployment: save the model to a file (uncomment to use)
# import joblib
# joblib.dump(model, 'iris_model.pkl')  # Load later with joblib.load()
```

#### Detailed Explanations of the Steps

Let me break this down further so it's crystal clear why we do each part.

- **Loading and Exploring Data**: We start by loading – no web scraping needed here. Printing head/describe helps spot issues (e.g., outliers). The pairplot is key: it shows Setosa is easy to separate, but Versicolor/Virginica overlap on some features, so the model has to learn subtle differences.
  
- **Splitting and Scaling**: Train/test split prevents overfitting (model memorizing instead of generalizing). Stratify ensures equal species in both sets. Scaling is crucial for distance-based algos like KNN – without it, a feature in cm vs mm could skew everything.

- **Training**: .fit() is the magic – for KNN, it just stores the data. Prediction is fast but happens at query time.

- **Evaluation**: Accuracy is simple but not always best (e.g., if imbalanced data). Classification report adds depth: precision (how many predicted Setosa are real?), recall (did we catch all Setosa?). Confusion matrix visually shows errors – ideally, diagonal is high, off-diagonal low.

- **Improvement**: This is hyperparameter tuning. k too small? Overfits noise. Too big? Underfits patterns. We loop and plot to find the sweet spot (often 3-7 here). In real projects, use GridSearchCV for automation.

**Typical Results**: On this split, accuracy hits 1.000 (100%) often, but that's because Iris is easy. In tougher datasets, 80-90% is good. If you rerun with different random_state, it might drop a bit – that's variability.

**Extensions to Try**:
- Swap KNN for LogisticRegression or DecisionTreeClassifier (import from sklearn).
- Add cross-validation: Use cross_val_score for more robust evaluation.
- Real data: Grab something from Kaggle, like Titanic survival prediction.
- Deployment: Wrap in a Flask app for a web interface (input measurements, get species).
## Conclusion
Machine Learning is more than just dry lines of code; it is the mindset of transforming raw data into real-world value. Hopefully, this article has equipped you with the solid foundation needed to confidently build your first model. So, fire up your computer and start practicing today—because the journey to mastering AI always begins with the simplest steps.
## Refference 
- (2026). Mindlabinc.ca. https://mindlabinc.ca/wp-content/uploads/2024/05/Machine-Learning.webp
- Sarker, I. H. (2021). Machine Learning: Algorithms, Real-World Applications and Research Directions. SN Computer Science, 2(3), 1–21. Springer. https://link.springer.com/article/10.1007/s42979-021-00592-x

‌