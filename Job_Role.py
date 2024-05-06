from Extraction import extract_resume_features
from Prediction_Job_Role import prediction

def predict_job_role(path):
    skills, sb, dg, exp= extract_resume_features(path)
    role= prediction(skills, sb, dg, exp)
    return role