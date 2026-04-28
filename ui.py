import pandas as pd
import streamlit as st
from tree_model import Model
from tree_model import categorical_string_values
from tree_model import x,scaler
from tree_model import ShowImage
st.title("**Credit Risk Analysis Project**")
st.write("🔒 YASAL UYARI / LEGAL DISCLAIMER Bu yazılım, yalnızca eğitim ve öğrenme amaçlı geliştirilmiş bir projedir.Türkiye Cumhuriyeti Anayasası'nın 20. maddesi gereğince herkes, özel hayatına ve aile hayatına saygı gösterilmesini isteme hakkına sahiptir. Bu proje, hiçbir gerçek kişi veya kurumun kredi başvurusu değerlendirmesinde kullanılmamaktadır.")
st.write("Bilgilendirme:""Eğitim Amaçlıdır: Bu model, yapay zeka ve makine öğrenmesi öğrenimi sürecinde geliştirilmiş olup, ticari veya resmi hiçbir karar mekanizmasında kullanılmamaktadır.Gerçek Veri İşlemez: Projede kullanılan veriler anonimleştirilmiş, hiçbir gerçek kişisel veri (KVKK 6698 sayılı Kanun kapsamında) işlenmemektedir.Karar Destek Sistemi Değildir: Modelin çıktıları hiçbir resmi veya özel kurum tarafından kredi onay/red kararı olarak değerlendirilemez.Sorumluluk Reddi: Bu modelin sonuçlarına dayanarak alınacak hiçbir karardan geliştirici sorumlu değildir.")
st.write("Anayasa Dayanağı: Madde 20: Özel hayatın gizliliği Madde 22: Haberleşme özgürlüğü Madde 48: Çalışma ve sözleşme özgürlüğü")
check_box=st.checkbox("I understand , I accept that")
if check_box:
    st.write("**Welcome**")
    age=st.number_input("**Age :**",min_value=18,max_value=110,value=18)
    income=st.number_input("**Salary :**",min_value=620,max_value=100000)
    home_ownership=st.selectbox("**Home Ownership :**",options=['RENT','OWN','MORTGAGE','OTHER'])
    person_emp_length=st.number_input("**How many years have you working ?",min_value=0,max_value=45,value=1)
    loan_intent=st.selectbox("Why do you want credit ?",options=['PERSONAL','EDUCATION','MEDICAL','VENTURE','DEBTCONSOLIDATION',"HOMEIMPROVEMENT"])
    loan_grade=st.selectbox("What is your credit score ?",options=['A','B','C','D','E','F','G'])
    loan_amnt=st.number_input("**Loan Amount :**",min_value=1000,max_value=100000,value=1000)
    loan_int_rate=st.number_input("What is your loan credit rate",min_value=2,max_value=25,value=2)
    loan_percent_income=st.number_input("**Loan Percent Income :**",min_value=0.01,max_value=0.99)
    buton=st.button("Determine",use_container_width=True)
    new_customer = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_home_ownership': [home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income]
    })
    if buton:
        st.write("Your request is being calculated at the moment...")
        with st.spinner("🔄You have to wait for a second..."):
            model = Model()
            new_customer = pd.get_dummies(new_customer, columns=categorical_string_values, drop_first=True)
            new_customer = new_customer.reindex(columns=x.columns, fill_value=0)

            for col in x.columns:
                if col not in new_customer.columns:
                    new_customer[col] = 0

            new_customer = new_customer[x.columns]

            result = model.func_model(new_customer)

            if result[0]==[1]:
                st.success("You can take the credit")
            if result[0]==[0]:
                st.info("You can not take the credit")

    buton_three = st.button("Show Tree Image", key=15, use_container_width=True)
    if buton_three:
        result_two = ShowImage()
        new_customer = pd.get_dummies(new_customer, columns=categorical_string_values, drop_first=True)
        new_customer = new_customer.reindex(columns=x.columns, fill_value=0)

        for col in x.columns:
            if col not in new_customer.columns:
                new_customer[col] = 0

        new_customer = new_customer[x.columns]
        image = result_two.show_tree_image(new_customer)
        st.pyplot(image)

else:
    st.info("You have to accept contract!")
