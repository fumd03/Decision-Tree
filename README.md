                            Penguin Species Classifier

A machine learning implementation using a Decision Tree Classifier to identify penguin species based on morphological measurements and environmental data.
Project Overview

This repository contains a streamlined end-to-end pipeline for classifying penguins into three species: Adelie, Chinstrap, and Gentoo. By leveraging the scikit-learn framework, the project demonstrates how to handle mixed data types (numerical and categorical) through automated preprocessing pipelines.

Dataset Insights
The model is trained on physical measurements of penguins observed in the Palmer Archipelago, Antarctica.
Features

    Environmental: island
    Morphological:
        culmen_length_mm & culmen_depth_mm
        flipper_length_mm
        body_mass_g
        Demographic: sex

Target Labels
The classifier predicts one of the following species:
Adelie
Chinstrap
Gentoo
Model Architecture

The project utilizes a robust machine learning workflow to ensure data integrity and prevent leakage:
Algorithm: DecisionTreeClassifier
Preprocessing:
OneHotEncoder used for categorical variables (island, sex).
StandardScaler (optional/internal) for numerical consistency.
Pipeline: Built using scikit-learn Pipelines to encapsulate the preprocessing and modeling steps into a single, deployable object.

Setup & Installation
Ensure you have Python 3.8+ installed.
Clone the repository:
Bash
git clone https://github.com/your-username/DecisionTree-Penguins.git
cd DecisionTree-Penguins
Install dependencies:
Bash
pip install -r requirements.txt
Run the model:
Bash
python main.py
Tech Stack
Language: Python
ML Framework: scikit-learn
Data Manipulation: Pandas & NumPy
Visualization: Matplotlib / Seaborn (for tree visualization)
