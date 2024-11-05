import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Sample data for the skills JSON file (skills.json)
skills_data = {
    "technical_skills": {
        "Python": 9,
        "Java": 7,
        "JavaScript": 6,
        "React": 7,
        "TensorFlow": 8,
        "Machine Learning": 7,
        "Cloud Computing": 6
    },
    "soft_skills": {
        "Speaking": 7,
        "Writing": 8,
        "Negotiation": 6
    },
    "domain_critical_skills": {
        "dev": ["Java", "JavaScript", "React"],
        "ml": ["Python", "TensorFlow", "Machine Learning"],
        "cloud": ["Cloud Computing", "JavaScript"]
    }
}

# Save skills data to JSON file for reference
with open('skills.json', 'w') as f:
    json.dump(skills_data, f)

# Load the dataset
def load_data():
    # Sample dataset creation for demonstration purposes
    data = {
        "cgpa": [8.5, 7.0, 9.0, 6.5],
        "final_technical_score": [27.5, 20.0, 30.0, 15.0],
        "soft_skills_score": [18.0, 15.0, 20.0, 10.0],
        "aptitude_test_score": [85, 70, 90, 60],
        "internship_experience": [1, 0, 1, 0],
        "projects_relevant": [1, 0, 1, 0],
        "domain": ["ml", "dev", "ml", "cloud"],
        "placed": [1, 0, 1, 0]
    }
    return pd.DataFrame(data)

# Train the Random Forest model
def train_model(data):
    X = data[['cgpa', 'final_technical_score', 'soft_skills_score', 'aptitude_test_score', 
              'internship_experience', 'projects_relevant']]
    y = data['placed']
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'placement_model.pkl')

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

# Load the trained Random Forest model
def load_trained_model():
    return joblib.load('placement_model.pkl')

# Function to calculate technical skills score
def calculate_technical_skills_score(user_skills, user_domain):
    with open('skills.json') as f:
        skills_data = json.load(f)

    user_skills_list = [skill.strip() for skill in user_skills.split(',')]
    crucial_skills = skills_data['domain_critical_skills'].get(user_domain, [])
    
    technical_skills_score = 0
    crucial_skills_score = 0

    for skill in user_skills_list:
        skill_score = skills_data['technical_skills'].get(skill, 0)
        technical_skills_score += skill_score
        
        if skill in crucial_skills:
            crucial_skills_score += skill_score * 1.5  # Weight for critical skills

    final_technical_score = (technical_skills_score + crucial_skills_score) / 2
    return final_technical_score

# Function to recommend missing and low-scoring skills
def recommend_missing_skills(user_skills, user_domain):
    with open('skills.json') as f:
        skills_data = json.load(f)

    user_skills_list = [skill.strip() for skill in user_skills.split(',')]
    crucial_skills = skills_data['domain_critical_skills'].get(user_domain, [])
    
    missing_skills = [skill for skill in crucial_skills if skill not in user_skills_list]
    low_scoring_skills = []
    
    for skill in user_skills_list:
        skill_score = skills_data['technical_skills'].get(skill, 0)
        if skill in crucial_skills and skill_score < 7:  # Threshold for low score
            low_scoring_skills.append(skill)
    
    return missing_skills, low_scoring_skills

# Main function to take user input and predict placement
def main():
    # Load and train the model (comment this after the first run to avoid retraining)
    data = load_data()
    train_model(data)

    # Load the trained model
    model = load_trained_model()

    # Example user input
    user_input = {
        'cgpa': 8.5,
        'user_skills': "Python, TensorFlow, JavaScript",
        'soft_skills_score': 18.0,
        'aptitude_test_score': 85,
        'internship_experience': 1,
        'projects_relevant': 1,
        'user_domain': 'ml'  # Machine Learning domain
    }

    # Calculate technical skills score
    final_technical_score = calculate_technical_skills_score(user_input['user_skills'], user_input['user_domain'])

    # Predict placement
    input_features = [user_input['cgpa'], final_technical_score, user_input['soft_skills_score'], 
                     user_input['aptitude_test_score'], user_input['internship_experience'], 
                     user_input['projects_relevant']]
    
    placement_prediction = model.predict([input_features])[0]

    # Output placement prediction
    if placement_prediction == 1:
        print("User will likely be placed!")
    else:
        print("User is less likely to be placed.")

    # Recommend missing and low-scoring skills
    missing_skills, low_scoring_skills = recommend_missing_skills(user_input['user_skills'], user_input['user_domain'])

    print(f"Missing crucial skills: {missing_skills}")
    print(f"Skills that need improvement: {low_scoring_skills}")

if __name__ == "__main__":
    main()
