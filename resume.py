import Skills_Prediction
import Similarity_function
import Resume_Extraction
import Skills_Extraction
def process_resume(path, jd, role):
    similarity_score, str1, str2= Similarity_function.find_similarity(path,jd)
    resume_text= Resume_Extraction.resume_extraction(path)
    resume_skills= Skills_Extraction.skills_extraction(resume_text)
    jd_skills= Skills_Extraction.skills_extraction(jd)
    have_skills, not_have_skills= Skills_Prediction.skills_prediction(resume_skills, jd_skills,role)
    return similarity_score,str1,str2, have_skills, not_have_skills