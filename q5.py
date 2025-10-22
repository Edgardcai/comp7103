"""
COMP7103 Assignment 1 - Question 5
Classification in Python: Gallstone Dataset using scikit-learn
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_validate
import warnings

warnings.filterwarnings('ignore')


def prepare_dataset(csv_file='gallstone.csv'):
    """
    Question 5a: Prepare the dataset before building the decision tree.

    Reads gallstone.csv and prepares features and target variable.
    Uses only the 7 attributes specified in Question 4:
    - Gender, Comorbidity, CAD, Hypothyroidism, Hyperlipidemia, DM, HFA
    And uses "Gallstone Status" as the class label.

    Args:
        csv_file: Path to the gallstone.csv file

    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Select the 7 required attributes (features)
    feature_columns = [
        'Gender',
        'Comorbidity',
        'Coronary Artery Disease (CAD)',
        'Hypothyroidism',
        'Hyperlipidemia',
        'Diabetes Mellitus (DM)',
        'Hepatic Fat Accumulation (HFA)'
    ]

    # Class label
    target_column = 'Gallstone Status'

    # Extract features (X) and target (y)
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Create feature names for display
    feature_names = [
        'Gender',
        'Comorbidity',
        'CAD',
        'Hypothyroidism',
        'Hyperlipidemia',
        'DM',
        'HFA'
    ]

    # Rename columns for better display
    X.columns = feature_names

    print("Dataset prepared successfully!")
    print(f"Number of instances: {len(X)}")
    print(f"Number of features: {X.shape[1]}")#X.shape 返回一个元组，表示矩阵X的维度，例如 (319, 7) 表示有319行和7列
    print(f"Features: {list(feature_names)}")
    print(f"\nFeature value ranges:")
    for col in X.columns:
        print(f"  {col}: {sorted(X[col].unique())}")
    print(f"\nTarget (Gallstone Status) values: {sorted(y.unique())}")
    print(f"Class distribution: {dict(y.value_counts().sort_index())}")

    return X, y, feature_names


def build_and_visualize_tree(X, y, feature_names):
    """
    Question 5b: Build and visualize the decision tree in text form.

    Builds a DecisionTreeClassifier with max_leaf_nodes=7 and visualizes it.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names

    Returns:
        model: Trained DecisionTreeClassifier
    """
    # Build decision tree with max 7 leaf nodes
    model = DecisionTreeClassifier(
        max_leaf_nodes=7,
        random_state=42  # For reproducibility
    )

    # Train on all data
    model.fit(X, y)

    print("\nDecision tree built successfully!")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    print(f"Number of nodes: {model.tree_.node_count}")

    # Visualize the tree in text form
    tree_rules = export_text(model, feature_names=feature_names)

    return model, tree_rules


def cross_validation_evaluation(model, X, y):
    """
    Question 5d: Perform cross-validation and calculate average accuracy.

    Performs 10-fold cross-validation and computes accuracy for each fold.

    Args:
        model: The DecisionTreeClassifier model
        X: Feature matrix
        y: Target vector

    Returns:
        cv_results: Dictionary containing cross-validation results
    """
    # Perform 10-fold cross-validation
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=10,  # 10-fold cross-validation
        scoring='accuracy',
        return_train_score=False
    )

    # Extract accuracy scores
    accuracy_scores = cv_results['test_score']

    print("\n10-Fold Cross-Validation Results:")
    print("=" * 50)
    for fold, score in enumerate(accuracy_scores, 1):
        print(f"Fold {fold:2d}: Accuracy = {score:.6f} ({score*100:.2f}%)")

    print("=" * 50)
    print(f"Average Accuracy: {accuracy_scores.mean():.6f} ({accuracy_scores.mean()*100:.2f}%)")
    print(f"Standard Deviation: {accuracy_scores.std():.6f}")

    return cv_results


def main():
    """
    Main function to execute all parts of Question 5.
    """
    print("=" * 70)
    print("COMP7103 Assignment 1 - Question 5")
    print("Classification in Python using scikit-learn")
    print("=" * 70)

    # Check if CSV file exists
    csv_file = 'gallstone.csv'
    try:
        # Question 5a: Prepare the dataset
        print("\n" + "=" * 70)
        print("Question 5a: Preparing the dataset")
        print("=" * 70)
        X, y, feature_names = prepare_dataset(csv_file)

        # Question 5b: Build and visualize the decision tree
        print("\n" + "=" * 70)
        print("Question 5b: Building and visualizing the decision tree")
        print("=" * 70)
        model, tree_rules = build_and_visualize_tree(X, y, feature_names)

        # Question 5c: Display the visualized decision tree
        print("\n" + "=" * 70)
        print("Question 5c: Visualized Decision Tree (text form)")
        print("=" * 70)
        print(tree_rules)

        # Question 5d: Cross-validation
        print("\n" + "=" * 70)
        print("Question 5d: Cross-validation evaluation")
        print("=" * 70)
        cv_results = cross_validation_evaluation(model, X, y)

        # Question 5e: Display results summary
        print("\n" + "=" * 70)
        print("Question 5e: Summary of Results")
        print("=" * 70)
        print("\nAccuracy scores for each fold:")
        accuracy_scores = cv_results['test_score']
        for fold, score in enumerate(accuracy_scores, 1):
            print(f"  Fold {fold:2d}: {score:.6f}")
        print(f"\nOverall Average Accuracy: {accuracy_scores.mean():.6f}")

        print("\n" + "=" * 70)
        print("All tasks completed successfully!")
        print("=" * 70)

    except FileNotFoundError:
        print(f"\nError: {csv_file} not found in current directory.")
        print("Please ensure gallstone.csv exists in the same directory as this script.")
        print("\nYou can generate it by running q4.py first, or it should have been")
        print("created automatically during the data preprocessing step.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# ============================================================================
# Code Segments for Assignment Submission
# ============================================================================

def show_code_segments():
    """
    Display the code segments as required for the assignment submission.
    """

    print("\n" + "=" * 70)
    print("CODE SEGMENTS FOR ASSIGNMENT SUBMISSION")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("Question 5a: Code segment to prepare the dataset")
    print("-" * 70)
    print("""
import pandas as pd

# Read the CSV file
df = pd.read_csv('gallstone.csv')

# Select the 7 required attributes (features)
feature_columns = [
    'Gender',
    'Comorbidity',
    'Coronary Artery Disease (CAD)',
    'Hypothyroidism',
    'Hyperlipidemia',
    'Diabetes Mellitus (DM)',
    'Hepatic Fat Accumulation (HFA)'
]

# Class label
target_column = 'Gallstone Status'

# Extract features (X) and target (y)
X = df[feature_columns].copy()
y = df[target_column].copy()

# Rename columns for better display
X.columns = ['Gender', 'Comorbidity', 'CAD', 'Hypothyroidism',
             'Hyperlipidemia', 'DM', 'HFA']
""")

    print("\n" + "-" * 70)
    print("Question 5b: Code segment to build and visualize the decision tree")
    print("-" * 70)
    print("""
from sklearn.tree import DecisionTreeClassifier, export_text

# Build decision tree with max 7 leaf nodes
model = DecisionTreeClassifier(max_leaf_nodes=7, random_state=42)

# Train on all data
model.fit(X, y)

# Visualize the tree in text form
feature_names = ['Gender', 'Comorbidity', 'CAD', 'Hypothyroidism',
                 'Hyperlipidemia', 'DM', 'HFA']
tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)
""")

    print("\n" + "-" * 70)
    print("Question 5d: Code segment for cross-validation")
    print("-" * 70)
    print("""
from sklearn.model_selection import cross_validate

# Perform 10-fold cross-validation
cv_results = cross_validate(
    model,
    X,
    y,
    cv=10,
    scoring='accuracy',
    return_train_score=False
)

# Extract and display accuracy scores
accuracy_scores = cv_results['test_score']
average_accuracy = accuracy_scores.mean()

print("Accuracy scores for each fold:")
for fold, score in enumerate(accuracy_scores, 1):
    print(f"Fold {fold}: {score:.6f}")
print(f"\\nAverage Accuracy: {average_accuracy:.6f}")
""")

# Uncomment the line below to display the code segments
show_code_segments()
