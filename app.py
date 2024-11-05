import json
from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__)

# Load the trained classification models and scaler
clf_model = joblib.load('placement_model.pkl')          # Logistic Regression
# Random Forest Classifier
rf_clf_model = joblib.load('rf_placement_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the trained regression model
reg_model = joblib.load('salary_model.pkl')

# Load skills data


def load_skills_data():
    with open('skills.json', 'r') as file:
        skills_data = json.load(file)
    return skills_data


skills_data = load_skills_data()


def calculate_technical_skills_score(user_skills, user_domain, projects):
    # Parse user skills
    user_skills_list = [skill.strip().lower()
                        for skill in user_skills.split(',')]

    # Get the crucial skills for the user's domain
    crucial_skills = skills_data['domain_critical_skills'].get(user_domain, [])

    technical_skills_score = 0
    crucial_skills_score = 0

    # Extract project skills
    project_skills = set()
    for project in projects:
        for skill in project['skills']:
            project_skills.add(skill.strip().lower())

    for skill in user_skills_list:
        # Add to total technical score
        skill_score = skills_data['technical_skills'].get(skill, 0)

        # Check if the skill is crucial for the selected domain
        if skill in crucial_skills:
            # Only add increased weight if the skill is also mentioned in projects
            if skill in project_skills:
                # Increased weight for domain-critical skills
                crucial_skills_score += skill_score * 1.5
                print("CRUTIAL")
            else:
                crucial_skills_score += skill_score
                print("NOT CRUTIAL")

    # Final technical score, with extra weight given to crucial skills mentioned in projects
    final_technical_score = crucial_skills_score

    # Convert the final technical score to a percentage
    selected_domain_skills = skills_data['domain_critical_skills'][user_domain]
    max_possible_score = sum(
        skills_data['technical_skills'][skill] for skill in selected_domain_skills) * 1.5
    
    final_technical_score_percentage = (
        final_technical_score / max_possible_score) * 100

    return final_technical_score_percentage


def calculate_soft_skills_score(user_soft_skills):
    # Parse user soft skills
    user_soft_skills_list = [skill.strip().lower()
                             for skill in user_soft_skills.split(',')]

    soft_skills_score = 0

    for skill in user_soft_skills_list:
        # Add soft skills scores based on the predefined weights
        skill_score = skills_data['soft_skills'].get(skill, 0)
        soft_skills_score += skill_score

    return soft_skills_score


def recommend_missing_skills(user_skills, user_domain, projects):
    user_skills_set = set([skill.strip().lower()
                          for skill in user_skills.split(',')])
    crucial_skills = set([skill.lower(
    ) for skill in skills_data['domain_critical_skills'].get(user_domain, [])])

    missing_skills = list(crucial_skills - user_skills_set)

    project_skills_set = set()
    for project in projects:
        for skill in project['skills']:
            project_skills_set.add(skill.strip().lower())

    skills_not_in_projects = list(user_skills_set - project_skills_set)

    return missing_skills, skills_not_in_projects


def recommend_missing_soft_skills(user_soft_skills):
    user_soft_skills_set = set([skill.strip().lower()
                               for skill in user_soft_skills.split(',')])
    all_soft_skills = set([skill.lower()
                          for skill in skills_data['soft_skills'].keys()])

    missing_soft_skills = list(all_soft_skills - user_soft_skills_set)

    return missing_soft_skills


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Retrieve form data
            user_name = request.form['user_name']
            roll_number = request.form['roll_number']
            cgpa = float(request.form['cgpa'])
            user_skills = request.form['user_skills']
            user_soft_skills = request.form['soft_skills']
            aptitude_test_score = float(request.form['aptitude_test_score'])
            internship_experience = 1 if request.form.get(
                'internship_experience') == 'yes' else 0
            user_domain = request.form['user_domain'].lower()

            # Retrieve projects data
            projects = []
            project_titles = request.form.getlist('project_title')
            project_skills = request.form.getlist('project_skills')

            for title, skills in zip(project_titles, project_skills):
                projects.append({
                    'title': title,
                    'skills': [skill.strip().lower() for skill in skills.split(',')]
                })

            # Calculate technical skills score with projects
            final_technical_score = calculate_technical_skills_score(
                user_skills, user_domain, projects
            )

            print("final_technical_score", final_technical_score)
            # Calculate soft skills score
            soft_skills_score = calculate_soft_skills_score(user_soft_skills)

            # Prepare input features for the model
            input_features = [
                cgpa,
                final_technical_score,
                soft_skills_score,
                aptitude_test_score,
                internship_experience,
                1 if user_domain == 'ml' else 0,
                1 if user_domain == 'dev' else 0,
                1 if user_domain == 'cloud' else 0
            ]

            # Ensure feature order matches the training phase
            feature_order = ['cgpa', 'final_technical_score', 'soft_skills_score',
                             'aptitude_test_score', 'internship_experience',
                             'domain_ml', 'domain_dev', 'domain_cloud']

            print("Input Features:", input_features)

            # Scale the features using the loaded scaler
            input_features_scaled = scaler.transform([input_features])
            print("Scaled Input Features:", input_features_scaled)

            # Predict placement
            placement_prediction = clf_model.predict(input_features_scaled)[0]
            placement_proba = clf_model.predict_proba(input_features_scaled)[0]

            print(
                f"Placement Prediction: {'Placed' if placement_prediction == 1 else 'Not Placed'}")
            print(f"Prediction Probabilities: {placement_proba}")

            # Predict estimated salary
            if placement_prediction == 1:
                estimated_package = reg_model.predict(input_features_scaled)[0]
                estimated_package = round(estimated_package, 2)
            else:
                estimated_package = 0

            print(f"Estimated Salary: {estimated_package}")

            # Recommend missing and low-scoring skills
            missing_skills, skills_not_in_projects = recommend_missing_skills(
                user_skills, user_domain, projects)
            missing_soft_skills = recommend_missing_soft_skills(
                user_soft_skills)

            return render_template('result.html',
                                   user_name=user_name,
                                   roll_number=roll_number,
                                   placement=placement_prediction,
                                   estimated_package=estimated_package,
                                   missing_skills=missing_skills,
                                   low_scoring_skills=skills_not_in_projects,
                                   missing_soft_skills=missing_soft_skills,
                                   projects=projects)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', error="An error occurred during processing. Please check your inputs.")
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, port=3000)
