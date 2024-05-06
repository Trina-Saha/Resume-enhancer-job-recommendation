import streamlit as st
from Job_Role import predict_job_role
import mysql.connector
import random
import resume
import base64
import os
from streamlit_tags import st_tags

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def fetch_job_info(predicted_role, num_links, work_location, work_type):
    # Establish MySQL connection
    mysql_conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='job_database'
    )
    
    # Create a cursor object to execute SQL queries
    cursor = mysql_conn.cursor()
    
    # Construct the SQL query based on the selected work location and work type
    if work_location == "Default" or work_type == "None":
        query = f"SELECT Company, JobLink FROM links_table WHERE Role = '{predicted_role}' LIMIT {num_links}"
    else:
        query = f"SELECT Company, JobLink FROM links_table WHERE Role = '{predicted_role}' AND Location = '{work_location}' AND WorkType = '{work_type}' LIMIT {num_links}"
    
    # Execute the SQL query
    cursor.execute(query)
    
    # Fetch the job links and company names from the database
    all_job_info = cursor.fetchall()
    
    # Shuffle the list of job links and select the specified number of links
    random.shuffle(all_job_info)
    selected_job_info = all_job_info[:num_links]
    
    # Close the cursor and MySQL connection
    cursor.close()
    mysql_conn.close()
    
    return selected_job_info

def job_recommendation_page():
    st.title("Personalized Recommendations for Your Professional Success")
    uploaded_file = st.file_uploader("**Upload your resume**", type=["pdf"])
    
    if uploaded_file is not None:
        num_links = st.text_input("**How many jobs you want to see?**", "")
        num_links = int(num_links) if num_links else None  # Convert input to integer if not empty
        
        # Add dropdowns for work location and work type
        col1, col2 = st.columns(2)
        with col1:
            work_location = st.selectbox("**Work Location**", ["Default", "Kolkata", "Bangalore", "Gurgaon", "Chennai", "Mumbai", "Hyderabad", "Noida", "Pune"])
        with col2:
            work_type = st.selectbox("**Work Type**", ["None", "Intern", "Full-Time", "Part-Time", "Contract", "Temporary"])
        
        submit_button = st.button("**Submit**")
        
        if submit_button:
            # Save the uploaded file
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Call the function to predict job role (assuming predict_job_role function is defined elsewhere)
            predicted_role = predict_job_role(file_path)
            
            # Display the predicted job role
            st.markdown(f'''<h4 style='text-align: left;'>You are fit for the role: {predicted_role.upper()}</h4>''', unsafe_allow_html=True)
            
            
            # Fetch job links and company names for the predicted role, work location, and work type
            job_info = fetch_job_info(predicted_role, num_links, work_location, work_type)
            
            # Display job links and company names on the webpage
            st.markdown('''<h4 style='text-align: left; color: black; '>Job Links</h4>''', unsafe_allow_html=True)
           
            for company, link in job_info:
                st.markdown(f"{company} [Click here to apply]({link})")

def resume_enhancer_page():
    st.title('Enhance your resume to get your desired jobs')

    # Upload resume
    uploaded_resume = st.file_uploader("**Upload your resume**", type=["pdf", "txt", "docx"])

    # Input for job description
    job_description = st.text_area("**Job Description**")

    # Dropdown for job roles
    job_roles = ["Accountant", "HR", "Web Designer", "Database Administrator", "Business Development Analyst", "Software Developer", "Data Scientist", "Nurse","Sales","Therapist"]
    selected_job_role = st.selectbox("**Select Job Role**", job_roles)

    # Submit button
    submit_button = st.button("**Submit**")

    if submit_button:
        # Save the uploaded file
        file_path = os.path.join("uploads", uploaded_resume.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())
        if uploaded_resume is None or job_description == "":
            st.error("Please upload a resume and specify job description.")
        else:
            # Display uploaded resume using show_pdf function
            if file_path:
                show_pdf(file_path)
            st.header("**Here are your results!**")
            # Call resume.py function to get similarity score, resume skills, and required skills
            similarity_score,edu,exp, resume_skills, required_skills = resume.process_resume(file_path, job_description, selected_job_role)

            # Display the outputs
            if similarity_score >= 50:
                st.markdown(f'''<h4 style='text-align: left; color: #008000;'>Your resume has {similarity_score}% chances of getting shortlisted.</h4>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<h4 style='text-align: left; color: #FF0000;'>Your resume has {similarity_score}% chances of getting shortlisted.</h4>''', unsafe_allow_html=True)
                if (edu!="" or exp!=""):
                    st.markdown(f'''<h6 style='font-weight: bold;text-align: left; color: black;'>Warnings:</h6>''', unsafe_allow_html=True)
                if edu!= '':
                    print(edu)  
                    st.markdown(f'''<h7 style='font-weight: bold;text-align: left; color: brown;'>{edu}</h7>''', unsafe_allow_html=True)
                if exp!= '':
                    print(exp)
                    st.markdown(f'''<h7 style='font-weight: bold;text-align: left; color: brown;'>{exp}</h7>''', unsafe_allow_html=True)
            # Display resume skills
            st.markdown('''<h4 style='text-align: left; color: black; '>Recommended skills you have</h4>''', unsafe_allow_html=True)
            if resume_skills:
                for skill in resume_skills:
                    st.write(skill.upper())
            else:
                st.write("Resume does not contain any predicted skills")

            # Display recommended skills
            st.markdown('''<h4 style='text-align: left; color: black;'>Enhance your job prospect with these skills</h4>''', unsafe_allow_html=True)
            for skill in required_skills:
                st.write(skill.upper())


# Main function to run the Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Resume Enhancer", "Job Recommendation"])

    if page == "Resume Enhancer":
        resume_enhancer_page()
    elif page == "Job Recommendation":
        job_recommendation_page()

if __name__ == "__main__":
    main()


