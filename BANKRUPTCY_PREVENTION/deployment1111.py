
import streamlit as st
import pickle 

with open('model_svm.pkl', 'rb') as load:
    model = pickle.load(load)

st.title('Bankruptcy Prevention')
st.image("https://images.app.goo.gl/52M4Kdh5yQQeyX2z6",width=300)


def predict(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk):
    prediction=model.predict([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])
    return prediction

def main():
    st.markdown("## ENTER THE DETAILS")

    # Input fields
    industrial_risk = st.number_input('Industrial Risk:', min_value=0.0, max_value=10.0, step=0.5)
    management_risk = st.number_input('Management Risk:', min_value=0.0, max_value=10.0, step=0.5)
    financial_flexibility = st.number_input('Financial Flexibility:', min_value=0.0, max_value=10.0, step=0.5)
    credibility = st.number_input('Credibility:', min_value=0.0, max_value=10.0, step=0.5)
    competitiveness = st.number_input('Competitiveness:', min_value=0.0, max_value=10.0, step=0.5)
    operating_risk = st.number_input('Operating Risk:', min_value=0.0, max_value=10.0, step=0.5)
    
    st.markdown('## CHECK THE VALUES')
    
    if industrial_risk!='':
        st.markdown(
            f'''
            *INDUSTRIAL RISK : {industrial_risk}  
            *MANAGEMENT RISK : {management_risk}  
            *FINANCIAL FLEXIBILITY : {financial_flexibility}  
            *CREDIBILITY : {credibility}  
            *COMPETITIVENESS : {competitiveness}  
            *OPERATING_RISK : {operating_risk}  
            ''')
            
    
    if st.button('Predict'):
            result = predict(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk)
            if result[0] == 0:  # Assuming the model returns [0] for Bankruptcy
                st.markdown('<p style="color:red;">Prediction: Bankruptcy</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:green;">Prediction: Non-Bankruptcy</p>',unsafe_allow_html=True)

if __name__ == '__main__':
    main()


