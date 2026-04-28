Link of Application : https://credit-risk-analysis-fyuybbjbov52kdrbqec3rz.streamlit.app/

🚀 Credit Risk Analysis Project
This project is a functional machine learning prototype developed as a term project for the Discrete Mathematics course. It utilizes a Decision Tree algorithm to predict whether a credit application should be approved or denied based on user input.

The application is currently managed via Git and deployed on Streamlit Cloud.


🧐 About the Project (Discrete Mathematics Connection)
The core of this project is based on Tree Structures and Graph Theory, which are fundamental topics in Discrete Mathematics.

Decision Trees: A Decision Tree is essentially a directed acyclic graph where each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label (Approval/Rejection).

Logical Operations: The model functions through a series of binary splits, mimicking the logical propositions and predicates studied in the course.



🛠️ Technical Stack
Python: Primary programming language.

Streamlit: Used for building the interactive web interface.

Scikit-Learn: Implements the DecisionTreeClassifier and GridSearchCV for model optimization.

Pandas: Used for data manipulation and preprocessing (One-Hot Encoding).

Matplotlib: Used to visualize the mathematical tree structure of the model.

Dotenv: Manages environment variables and file paths.


📂 Project Structure
ui.py: Handles the user interface, input validation (age, salary, loan intent, etc.), and displays the results.

tree_model.py: Contains the backend logic using Object-Oriented Programming (OOP). It handles data cleaning, scaling with StandardScaler, and the training of the Decision Tree model.




📊 Features
Automated Preprocessing: Automatically handles missing values and encodes categorical strings like 'Home Ownership' and 'Loan Intent'.

Hyperparameter Tuning: Uses GridSearchCV to find the most accurate tree depth and splitting criteria.

Visual Decision Path: Users can generate a visual representation of the Decision Tree to see exactly how the "Discrete Structures" logic is applied to their data.



⚖️ Legal Disclaimer
This software is developed for educational purposes only.

In accordance with Article 20 of the Constitution of the Republic of Turkey, it respects the privacy of private life.

No real personal data is processed (compliant with KVKK).

The outputs of this model are not financial advice and cannot be used by institutions for official credit decisions.




