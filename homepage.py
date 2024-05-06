import streamlit as st
import mysql.connector
import subprocess

mydb = mysql.connector.connect(
     host="localhost",   # Change this to your database server host
     user="root",        # Change this to your database username
     password="",        # Change this to your database password
     database="registered_user"  # Change this to your database name
    )
mycursor = mydb.cursor()
print("connection made")

def main():
    st.title("Welcome!")

    option = st.sidebar.selectbox("Connect with us", ("Login", "Register"))
    if option == "Register":
        st.subheader("Create your profile")
        name = st.text_input("Enter name")
        email = st.text_input("Enter Email")
        phone = st.text_input("Mobile Number")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            if not (name and email and phone and password):
                st.error('Please fill all the required fields')
                st.stop()
            else:
                sql = "INSERT INTO reg_user(NAME, EMAIL, PHONE, PASSWORD) VALUES (%s, %s, %s, %s)"
                val = (name, email, phone, password)
                mycursor.execute(sql, val)
                mydb.commit()
                st.success("Registration successful.")

    elif option == "Login":
        st.subheader("login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if not (username and password):
                st.error('Please fill all the required fields')
            else:
                mycursor.execute("SELECT NAME, PASSWORD FROM reg_user WHERE NAME=%s AND PASSWORD=%s", (username, password))
                user = mycursor.fetchone()
                mydb.close()
                if user is not None:
                    st.session_state.logged_in = True
                    st.success("Login successful")
                    subprocess.Popen(["streamlit", "run", "option_page3.py"])
                else:
                    st.error("Invalid username or password")

if __name__ == "__main__":
    main()
