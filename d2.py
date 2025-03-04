import streamlit as st
import pandas as pd
import numpy as np

excel_file = pd.ExcelFile("T1.xlsx")

images = r"D:\SREC\Dataset"

# Get all sheet names
sheet_names = excel_file.sheet_names
# print(sheet_names)
# df = pd.read_excel("T1.xlsx", sheet_name="Sheet2")
# print(df)

st.set_page_config(
    page_title="Employee Attendance Dashboard",
    page_icon="🧑‍🏭",
    layout="wide",
)

with st.sidebar:
    st.title('🧑‍🏭 Employee Attendance Dashboard 🧑‍🏭')
    selected_month = st.selectbox('Select a year', sheet_names)
    month = pd.read_excel("T1.xlsx",sheet_name=selected_month)
    employee_names = month['Name'].tolist()
    selected_employee = st.selectbox('Select an employee', employee_names)
    
col1, col2 = st.columns(2)

with col1:
    st.title(selected_employee)
    person_image_1 = images + "\\" + selected_employee + "\\" + selected_employee+"_0"+".jpg"
    st.image(person_image_1, caption=f"Employee ID : {id}", use_container_width=True)
    
    chart_data = pd.DataFrame(np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

    st.line_chart(chart_data)
    
with col2:
    st.write(f'## Attendance Summary {selected_month}')
    work = month.loc[month["Name"] == selected_employee, "Work"].values[0]
    ot = month.loc[month["Name"] == selected_employee, "OT"].values[0]
    st.metric(label="Work", value=work, delta="Hrs", delta_color="normal")
    st.metric(label="OT", value=ot, delta="Hrs")
    print(work)

