import os
import requests
import json
import joblib
from datetime import datetime
from io import BytesIO
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# func to get current datetime
def get_current_datetime():
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# config for the app
st.set_page_config(
    page_title="Housing Law Insight Dashboard",
    page_icon=":judge:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Housing Law Insight Dashboard :house_with_garden: :judge: :bar_chart:")

def init_session_states():
    """Initialize session state variables."""
    st.session_state['form_data_as_model_example'] = st.session_state.get('form_data_as_model_example', None)
    st.session_state['form_data'] = st.session_state.get('form_data', None)
    st.session_state['model_inference'] = st.session_state.get('model_inference', None)

# Initialize session state variables
init_session_states()

@st.cache_resource
def load_model_and_selector():
    """Load the trained model and feature selector from the cloud."""
    model_url = os.environ.get('MODEL_URL', None)
    feat_sel_url = os.environ.get('FEATURE_SELECTOR_URL', None)

    # Download and load the model
    model_response = requests.get(model_url)
    model_response.raise_for_status()
    best_model = joblib.load(BytesIO(model_response.content))

    # Download and load the feature selector
    selector_response = requests.get(feat_sel_url)
    selector_response.raise_for_status()
    selector = joblib.load(BytesIO(selector_response.content))

    return best_model, selector

# Function to display the About tab
def show_about():
    """App page explaining the project and its goals."""
    # st.markdown("## About this Project")
    st.markdown("## About this Project")

    ###### 'ABOUT' PAGE WRITEUP ######

    about_1 = """
    This web app is a demonstration project built to showcase the power of Data Science and Natural Language Processing (NLP) in revolutionizing access to justice. 

    It aims to understand the factors influencing eviction decisions in Ontario, Canada. This project involved analyzing over 50,000 case files, each containing a wealth of textual information about the circumstances surrounding eviction proceedings. 

    To extract meaningful insights from this vast dataset, the team employed cutting-edge NLP techniques, including a fine-tuned OpenAI "GPT-4o-mini" model, which was specifically trained to identify and extract over **250** key pieces of information from each case file. 

    This information was then transformed into a structured dataset, allowing the team to apply machine learning algorithms to identify patterns and correlations between various factors and the ultimate outcome of the case. 

    This web app aims to allow you to explore the types of data collected and understand how it can be used to gain valuable insights into legal decisions.

    **Here's a glimpse into the process:**

    1. **Data Extraction:** The app takes a case file as input. It then uses the trained NLP model to extract information about various aspects of the case, such as the landlord's representation, the tenant's financial situation, and the history of late payments. 
    2. **Data Visualization:** The extracted information is then presented to you in a structured format, allowing you to easily visualize the key details of the case. 
    3. **Model Inference:** A pre-trained machine learning model is then used to analyze this structured data and predict the likelihood of the case resulting in eviction. This prediction is based on the patterns learned from the analysis of thousands of real eviction cases. 

    **While this app is a simplified demonstration, it highlights the potential of data-driven approaches to:**

    * **Increase transparency and understanding of legal decisions:** By analyzing large datasets of legal documents, we can gain valuable insights into the factors that influence judicial outcomes. 
    * **Improve access to justice:** This knowledge can empower individuals and their legal representatives to better understand their rights and obligations, as well as to make more informed decisions during legal proceedings. 
    * **Promote fairness and equity:** By identifying tendencies in legal decisions, we can work towards a more interpretable and equitable legal system.
    """.replace("    ", "")
    st.markdown(about_1.strip())
    
    st.divider()

    about_2 = """
    ***Please note that this app is for educational and historical analysis purposes only and does not constitute the provision of any legal advice. Consult a legal professional for advice on specific legal matters.***
    """.replace("    ", "")
    st.markdown(about_2.strip())
    
    st.divider()

    about_3 = """
    We hope this app will inspire you to explore the exciting possibilities of data science and NLP in the legal domain.
    """.replace("    ", "")
    st.markdown(about_3.strip())

# Function to display the Form tab
def show_form():
    """App page for user to fill out the case information form."""

    # load an empty, collapsed sidebar
    st.markdown("## Case Information Form")
    st.markdown("*Please fill out the form below to the best of your ability. If you are unsure about any information, you can leave it blank or select 'Not Stated', or 'Not Applicable'.*")

    ### Having default values for continuous cols as mean of each column -- safest to assume if user is unsure
    continuous_cols = {
        'children_under_18' : 0, # mean was less than zero (mostly 'not stated')
        'children_under_14' : 0, # mean was less than zero (mostly 'not stated')
        'children_under_5' : 0, # mean was less than zero (mostly 'not stated')
        'tenancy_length' : 0, # user can answer this
        'monthly_rent' : 700, # mean is ~670 but we'll round up to 700
        'rental_deposit' : 550, # mean is ~550
        'post_increase_rent' : 0, # mean from data is ~40 but I don't want to skew the data
        'total_arrears' : 1000, # median is ~975 but we'll round up to 1000
        'arrears_duration' : 0, # mean from data is ~1 but we'll say 0 (immediate arrears payment)
        'payment_amount_post_notice' : 800, # mean is ~838 but we'll round down to 800
        'total_children' : 0, # mean was less than zero (mostly 'not stated')
        'household_income' : 0, # mean was more than zero but I don't want to skew anything here so leaving default to zero
        'payment_plan_length' : 0, # mean was less than zero (mostly 'not stated')
        'notice_duration' : 0, # mean was less than zero (mostly 'not stated')

        # these values can't be known by the user so we'll assume various values that are the most statistically likely given the data
        'payment_plan_proposed' : "Not Stated", # user can't know this so we'll assume this would be the most common value
        'payment_plan_accepted' : "Not Stated", # user can't know this so we'll assume this would be the most common value
        'payment_plan_length' : "Not Stated", # user can't know this so we'll assume this would be the most common value
        'postponement_leads_to_arrears' : "Not Stated", # user can't know this so we'll assume this would be the most common value
        'hearing_date_month' : 5, # most common month
        'hearing_date_day' : 14, # most common day
        'hearing_date_year' : get_current_datetime().split()[0].split('-')[0], # assume current year
        'decision_date_month' : 5, # most common month
        'decision_date_day' : 15, # most common day
        'decision_date_year' : get_current_datetime().split()[0].split('-')[0], # current year
        'hearing_decision_diff' : 16.204368086073647, # user can't know this so we'll assume this would be the dates difference
        'rent_increase_date_present' : "Not Stated", # user can't know this so we'll assume this would be the most common value
    }

    
    with st.form(key='case_info_form'):

        # encodings of the most common adjudicating members per location (that are not "Not stated") -- this doesn't need to be encoded downstream for model usage
        board_to_member_lookup = {
            'Barrie': 73,
            'Belleville': 94,
            'Bracebridge': 28,
            'Brantford': 53,
            'Brockville': 9,
            'Burlington': 121,
            'Chatham': 110,
            'Cobourg': 94,
            'Cochrane': 78,
            'Cornwall': 40,
            'Goderich': 11,
            'Guelph': 126,
            'Hamilton': 15,
            'Hawkesbury': 40,
            'Kawartha Lakes': 77,
            'Kingston': 9,
            'Kitchener': 49,
            'Lindsay': 94,
            'London': 11,
            'Mississauga': 73,
            'Newmarket': 76,
            'North Bay': 74,
            'North York': 59,
            'Orangeville': 25,
            'Oshawa': 94,
            'Ottawa': 9,
            'Owen Sound': 28,
            'Pembroke': 9,
            'Perth': 9,
            'Peterborough': 32,
            'Sarnia': 62,
            'Sault Ste. Marie': 78,
            'Severn': 97,
            'Simcoe': 38,
            'Smiths Falls': 106,
            'St. Catharines': 53,
            'St. Thomas': 107,
            'St.Thomas': 110,
            'Stratford': 131,
            'Sudbury': 74,
            'Thunder Bay': 74,
            'Timmins': 78,
            'Toronto': 26,
            'Waterloo': 110,
            'Whitby': 32,
            'Windsor': 70,
            'Woodstock': 107,
            'York': 193
        }

        # General Questions
        st.markdown("## General Information")
        board_location = st.selectbox(
            "What is the location of the board?",
            sorted(board_to_member_lookup.keys()),
        )

        adj_mem = board_to_member_lookup[board_location]

        cat_common_order_1 = ['Yes', 'No', 'Not Stated', 'Not Applicable']
        cat_common_order_2 = ['Not Applicable', 'No', 'Yes', 'Not Stated']
        cat_common_order_3 = ['Not Applicable', 'Yes', 'No', 'Not Stated']

        # Landlord Information
        st.markdown("## Information About the Landlord")

        landlord_represented = st.selectbox("Will the landlord be represented?", cat_common_order_1)
        landlord_attended_hearing = st.selectbox("Will the landlord attend the hearing?", cat_common_order_1)
        landlord_not_for_profit = st.selectbox("Is the landlord not-for-profit?", cat_common_order_1)

        # Tenant Information
        st.markdown("## Information About the Tenant")

        tenant_represented = st.selectbox("Will the tenant be represented?", cat_common_order_1)
        tenant_attended_hearing = st.selectbox("Will the tenant attend the hearing?", cat_common_order_1)
        tenant_conditions = st.selectbox(
            "Does the tenant have any conditions that would impact their life?",
            [
                'Not Applicable', 'Chronic Illnesses', 'Physical Disabilities',
                'Mental Health Issues', 'Access to Healthcare',
                'Employment and Financial Issues', 'Family and Dependent Care',
                'Environmental and Housing Issues', 'Other'
            ],
            help="Please list any conditions that the tenant has that may impact their life. For example, chronic illnesses, physical disabilities, mental health issues, etc."
        ) # this obviously needs better phrasing

        tenant_collecting_subsidy = st.selectbox("Is the tenant collecting a subsidy?", cat_common_order_1)
        tenancy_length = st.number_input("How long is/was the tenancy (in **months**)?", min_value=0, step=1)
        monthly_rent = st.number_input("What is/was the monthly rent?", min_value=0.0, step=0.01)
        rental_deposit = st.number_input("What is/was the rental deposit?", min_value=0.0, step=0.01)
        post_increase_rent = st.number_input("What is/was the rent after the increase?", min_value=0.0, step=0.01)
        
        # rent_increase_date = st.text_input("What is the rent increase date (YYYY-MM-DD or Not stated)?")

        # total_arrears = st.number_input("What is the total arrears?", min_value=0.0, step=0.01)
        arrears_duration = st.number_input("How long is/was the arrears duration (in months)?", min_value=0, step=1)
        payment_amount_post_notice = st.number_input("What is/was the payment amount post-notice?", min_value=0.0, step=0.01)

        history_of_arrears = st.selectbox("Is/was there a history of arrears?", cat_common_order_1)
        history_of_arrears_payments = st.selectbox("Is/was there a history of arrears payments?", cat_common_order_1)
        frequency_of_late_payments = st.selectbox("Is/was there a frequency of late payments?", cat_common_order_1)
        tenant_chose_not_to_pay = st.selectbox("Did the tenant choose not to pay?", cat_common_order_1)

        # Tenant Children Information
        st.markdown("## Tenant Children")

        tenant_has_children = st.selectbox("Does/did the tenant have children?", ['No', 'Yes', 'Not Stated', 'Not applicable'])
        total_children = st.number_input("How many children in total, regardless of age?", min_value=0, step=1)
        children_under_18 = st.number_input("How many children are under 18?", min_value=0, step=1)
        children_under_14 = st.number_input("How many children are under 14?", min_value=0, step=1)
        children_under_5 = st.number_input("How many children are under 5?", min_value=0, step=1)
        children_with_conditions = st.selectbox("Do/did any children have conditions?", cat_common_order_2)
        conditions_impact_on_moving = st.selectbox("Do/did the conditions impact moving?", cat_common_order_3)
        
        # Tenant Employment and Financial Information
        st.markdown("## Tenant Employment and Financial Situation")

        tenant_employed = st.selectbox("Is/was the tenant employed?", ['Yes', 'No', 'Not Stated'])
        tenant_receiving_assistance = st.selectbox("Is/was the tenant receiving financial assistance?", cat_common_order_1)
        employment_stability_concerns = st.selectbox("Are there concerns about employment stability?", ['Yes', 'No', 'Not Stated'])
        tenant_had_sufficient_income = st.selectbox("Does/did the tenant have sufficient income?", ['Yes', 'No', 'Not Stated'])

        household_income = st.number_input("What is/was the household income?", min_value=0.0, step=0.01)
        tenant_job_loss_during_period = st.selectbox("Does/did the tenant lose their job during this period?", cat_common_order_1)

        # Other Information
        st.markdown("## Other Information")
        other_extenuating_circumstances = st.selectbox("Are/were there other extenuating circumstances?", ['Not applicable', 'Health Issues', 'Family and Life Events', 'Employment and Financial Issues', 'Legal and Administrative Issues', 'Housing Conditions and Disputes', 'Other'])

        ### none of the following questions can be answered by the user so we'll assume the most common values
        payment_plan_proposed = st.selectbox("Could a payment plan be proposed?", cat_common_order_1)

        difficulty_finding_housing = st.selectbox("Is/was there difficulty finding housing?", cat_common_order_1)
        reasons_for_housing_difficulty = st.selectbox("What are/were the reasons for housing difficulty?", cat_common_order_1)

        tenant_given_prior_notice = st.selectbox("Was the tenant given prior notice?", cat_common_order_1)
        notice_duration = st.number_input("If the tenant was given prior notice, what is/was the duration of the notice (in days)?", min_value=0, step=1)
        valid_applications = [
            'W7', 'C19', 'E23', 'C1', 'G02', 'W3', 'G01', 'L7', 'C14', 'A22', 'N14', 'E33', 'R27', 'L85', 
            'D28', 'E10', 'K2', 'L4', 'S7', 'N5', 'C11', 'H7', 'C06', 'T2', 'W1', 'A7', 'F1', 'B10', 
            'G5', 'T6', 'C12', 'A23', 'C35', 'L35', 'R4', 'A1', 'T50', 'D7', 'A9', 'G1', 'C7', 'F4', 
            'A4', 'A20', 'N7', 'C2', 'E21', 'C22', 'L23', 'L10', 'J1', 'A12', 'N4', 'B37', 'I26', 'G6', 
            'B03', 'C6', 'D3', 'A6', 'L13', 'L2', 'T5', 'E2', 'H1', 'I2', 'B15', 'B3', 'E19', 'L1', 
            'N10', 'L5', 'P1', 'D2', 'A15', 'M3', 'A05', 'A35', 'N9', 'S2', 'R5', 'C01', 'N8', 'M2', 
            'B9', 'C27', 'D6', 'R20', 'A10', 'C34', 'B6', 'F10', 'R2', 'B2', 'L9', 'A3', 'L37', 'W4', 
            'E03', 'B7', 'C8', 'A01', 'A17', 'G12', 'B12', 'B19', 'L3', 'T44', 'T38', 'T49', 'N17', 'N3', 
            'N1', 'T7', 'W2', 'M4', 'E17', 'B28', 'T3', 'B01', 'D4', 'A16', 'L42', 'N2', 'H23', 'C36', 
            'T1', 'X12', 'D13', 'A36', 'L34', 'A09', 'C4', 'S82', 'A5', 'X7', 'B1', 'E41', 'T51', 'K9', 
            'A11', 'T4', 'W6', 'W8', 'S83', 'T48', 'G3', 'X18', 'N13', 'F3', 'D5', 'B8', 'L15', 'F14', 
            'L11', 'A2', 'R1', 'N6', 'T23', 'P38', 'E7', 'P2', 'B4', 'H19', 'E1', 'W5', 'P5', 'E16', 
            'T01', 'G2', 'J6', 'B11', 'N15', 'G69', 'B02', 'N0', 'E3', 'R32', 'N11', 'P4', 'R40', 'S10', 
            'L6', 'L12', 'B5', 'C5', 'N12', 'G4', 'C24', 'C21', 'P3', 'L8', 'L17', 'D1', 'C10', 'S17', 
            'J3', 'A8'
        ]

        # the below returns list of strings -- need to be parsed to columns
        extracted_applications = st.multiselect(
            label="Have any application forms been mentioned so far (example: L2, T2, etc.)?",
            options=sorted(valid_applications),
            placeholder="Select all that apply, or leave empty if not applicable",
            help="Please list any application forms mentioned in the case file. For more information on this topic, please visit the [Official Landlord & Tenant Board (LTB) website](https://tribunalsontario.ca/ltb/forms/)"
        )

        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            st.success("Form submitted successfully! Please proceed to the Results tab.")
            # Collect form data into a dictionary or use as needed
            form_data = {
                'adjudicating_member': adj_mem, # assumption because user can't know this
                'board_location': board_location,
                'landlord_represented': landlord_represented,
                'landlord_attended_hearing': landlord_attended_hearing,
                'tenant_represented': tenant_represented,
                'tenant_attended_hearing': tenant_attended_hearing,
                'landlord_not_for_profit': landlord_not_for_profit,
                'tenant_collecting_subsidy': tenant_collecting_subsidy,
                'tenancy_length': tenancy_length,
                'monthly_rent': monthly_rent,
                'rental_deposit': rental_deposit,
                'post_increase_rent': post_increase_rent,
                # 'rent_increase_date': rent_increase_date, # not used by model
                'rent_increase_date_present': continuous_cols['rent_increase_date_present'], # added in lieu of 'rent_increase_date'
                'total_arrears': continuous_cols['total_arrears'],
                'arrears_duration': arrears_duration,
                'payment_amount_post_notice': payment_amount_post_notice,
                'history_of_arrears': history_of_arrears,
                'history_of_arrears_payments': history_of_arrears_payments,
                'frequency_of_late_payments': frequency_of_late_payments,
                'tenant_chose_not_to_pay': tenant_chose_not_to_pay,
                'tenant_conditions': tenant_conditions,
                'tenant_has_children': tenant_has_children,
                'total_children': total_children,
                'children_under_18': children_under_18,
                'children_under_14': children_under_14,
                'children_under_5': children_under_5,
                'children_with_conditions': children_with_conditions,
                'conditions_impact_on_moving': conditions_impact_on_moving,
                'tenant_employed': tenant_employed,
                'tenant_receiving_assistance': tenant_receiving_assistance,
                'employment_stability_concerns': employment_stability_concerns,
                'tenant_had_sufficient_income': tenant_had_sufficient_income,
                'household_income': household_income,
                'tenant_job_loss_during_period': tenant_job_loss_during_period,
                'other_extenuating_circumstances': other_extenuating_circumstances,
                'payment_plan_proposed': payment_plan_proposed,
                'payment_plan_accepted': continuous_cols['payment_plan_accepted'],
                'payment_plan_length': continuous_cols['payment_plan_length'],
                'difficulty_finding_housing': difficulty_finding_housing,
                'reasons_for_housing_difficulty': reasons_for_housing_difficulty,
                'tenant_given_prior_notice': tenant_given_prior_notice,
                'notice_duration': notice_duration,
                'postponement_leads_to_arrears': continuous_cols['postponement_leads_to_arrears'],
                'n4_notice_validity': 'Not Stated', # assumption because user can't know this
                'extracted_applications': extracted_applications,
                'hearing_date_present': 'Not Stated', # assumption because user can't know this
                'decision_date_present': 'Not Stated', # assumption because user can't know this
                'hearing_date_month': continuous_cols['hearing_date_month'],
                'hearing_date_day': continuous_cols['hearing_date_day'],
                'hearing_date_year': continuous_cols['hearing_date_year'],
                'decision_date_month': continuous_cols['decision_date_month'],
                'decision_date_day': continuous_cols['decision_date_day'],
                'decision_date_year': continuous_cols['decision_date_year'],
                'hearing_decision_diff': continuous_cols['hearing_decision_diff'],
                # 'x_application_present': x_application_present,
            }

            # Save form data to session state
            st.session_state['form_data'] = form_data

            # read in col order from local txt file -- NOTE: THIS EXCLUDES 'case_outcome' COLUMN SINCE TRAINED MODEL DOESN'T TAKE IT AS INPUT
            with open('data/app-data/col_order.txt', 'r') as f:
                col_order = f.read().splitlines()

            application_cols = {col: "false" for col in col_order if "__application_present" in col}
            # update mentioned applications with True
            for app in extracted_applications:
                application_cols[f"{app}__application_present"] = "true"

            # order form data into a dict with the same order as col order, (excluding applications for now)
            form_data_as_model_example = {col: form_data[col] for col in col_order if col not in application_cols}

            # add applications to form data
            form_data_as_model_example.update(application_cols)

            # save example as session state
            st.session_state['form_data_as_model_example'] = form_data_as_model_example

            ########################### run trained model on submitted form data ###########################

            #### Placeholder for model inference results ####
            best_model, selector = load_model_and_selector()

            # # get the class labels
            class_labels = best_model.classes_

            # # use predict_proba to get probabilities from xgb model (loaded model) on form values
            LOOKUP = json.load(open('data/app-data/encoding-lookup.json', 'r'))
            lookup_classes = [val for val in LOOKUP['per_column']['case_outcome'].keys() if val != 'Not Stated']
            cols_to_encode = list(LOOKUP['per_column'].keys())
            if "adjudicating_member" in cols_to_encode:
                cols_to_encode.remove("adjudicating_member")

            # difference between the two lists
            already_encoded_cols = list(set(form_data_as_model_example.keys()) - set(cols_to_encode))
            if "adjudicating_member" not in already_encoded_cols:
                already_encoded_cols.append("adjudicating_member")

            encoded_form_data = []
            for col, val in form_data_as_model_example.items(): # need to do it in this order
                # check if col needs to be encoded, if so, encode, else just append the value
                try:
                    # if the column is in the "global" list ("Not Stated", "Not Applicable", etc.)
                    if val in LOOKUP['global']:
                        encoded_form_data.append(
                            int(LOOKUP['global'][val])
                        )
                    
                    # if the column is in the "to be encoded" list
                    elif col in cols_to_encode and val in LOOKUP['per_column'][col]:
                        encoded_form_data.append(
                            int(LOOKUP['per_column'][col][val])
                        )

                    # if the column is already encoded at form submission or does not require further encoding
                    else:
                        encoded_form_data.append(
                            int(val)
                        )
                except:
                    raise ValueError(f"Error with form data at column '{col}' with value '{val}'")

            # converting to 2D numpy array -- this is just what's expected by the feature selector
            encoded_form_data = np.array(encoded_form_data).reshape(1, -1)
            transformed_form_example = selector.transform(encoded_form_data)
            y_test_pred_proba = best_model.predict_proba(transformed_form_example)

            st.session_state['model_inference'] = {}
            for label, prob in zip(class_labels, y_test_pred_proba[0]):
                # Lookup the class name using the label index
                class_name = lookup_classes[label]
                # Store the class name and corresponding probability in the dictionary
                st.session_state['model_inference'][class_name] = float(prob) # converting from numpy float to python float

            # sort the results from most to least likely
            st.session_state['model_inference'] = dict(sorted(st.session_state['model_inference'].items(), key=lambda item: item[1], reverse=True))

            st.write(st.session_state['model_inference']) # this works fine...?

            #################################################################################################

# Function to display the Results tab
def show_results():
    """App page for displaying the model inference results."""
    st.write("Session State Contents:", st.session_state)
    if st.session_state.get('model_inference', None):
        st.markdown("## Results")
        st.write("## Model Inference Results")
        st.write("The model has made predictions based on the information you provided. Here are the results, and how to interpret them:")
        st.write("### Case Outcome Predictions")

        for outcome, prob in st.session_state['model_inference'].items():
            st.write(f"* **{outcome}**: {round(prob*100, 4)}")

        if st.button("Download Results"):
            st.write("File will be created to download")
    else:
        st.info("Please submit information via the Form tab.")

# Function to display the Data Exploration tab
def show_data_exploration():
    """App page for exploring the dataset."""
    st.markdown("## Explore the Data")
    st.write("This tab is under construction. Please check back later.")

def show_model_training():
    """App page explaining model training process"""
    st.markdown("## Model Training")

    mod_methods = """
    ### Methods

    The trained model in this project is a [XGBoost Classifier](https://xgboost.readthedocs.io/en/stable/) model. After testing other feature-based classifiers, including [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), the XGBoost model was chosen for the following reasons:
    1. Simply put, it performed the best. It achieved the highest *scores** in predicting the outcome of the cases in the test set.
    2. It is a robust model that can handle both categorical and continuous data, with the training data not required linear relationships (unlike a model like a linear regression classifier).
    3. Despite there being more modern model architectures like [Deep Learning Neural Networks](https://en.wikipedia.org/wiki/Deep_learning) and [Transformers](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)), this model still achieves great performance and is far more [interpretable](https://docs.aws.amazon.com/whitepapers/latest/model-explainability-aws-ai-ml/interpretability-versus-explainability.html). With the ultimate goal of this project being to get insights into the decision-making process of a Landlord-Tenant Board legal case, the trained model had to strike a balance of interpretability and performance.

    **See the **Training Results** section below for more details on the definition of 'scores' and 'performance' in the context of this project.*

    Broadly speaking, the model training process involved the following steps:
    1. **Data Preprocessing:** The dataset was cleaned and preprocessed to handle missing values, encode categorical variables, and scale numerical features.
    2. **Feature Engineering:** The dataset was transformed to create new features that could help the model learn patterns in the data.
    3. **Model Selection:** Various classifiers were tested, and the XGBoost model was chosen based on its performance metrics.
    4. **Hyperparameter Tuning:** The model's hyperparameters were optimized using techniques like Grid Search and Random Search to improve its performance.
    5. **Model Training:** The XGBoost model was trained on the preprocessed dataset to learn the patterns in the data.
    6. **Model Evaluation:** The trained model was evaluated on a separate test set to assess its performance in predicting the outcome of cases.
    7. **Model Interpretation:** The model's predictions were analyzed to understand the factors that influence eviction decisions in Ontario, Canada.
    """.replace("    ", "").strip()

    st.markdown(mod_methods)

    st.divider()

    st.markdown("### Testing the Trained Model")

    # sorted ascending
    categories = ['Full Eviction', 'Dismissal', 'Conditional Eviction', 'No Eviction, Payment Plan', 'Abatement', 'Relief From Eviction', 'Rent Adjustment', 'Postponement']
    baseline = [61.68, 13.36, 9.28, 8.2, 4.72, 1.87, 0.78, 0.11]
    best_model = [89.56, 72.26, 58.63, 58.72, 66.82, 27.59, 56.76, 66.67]
    ### proportion comparison in performance compared to baseline, rounded to 2 decimals
    comparison_to_baseline = [round((best_model[i] / baseline[i]), 2) for i in range(len(categories))]

    fig = go.Figure()

    # Add bars for baseline and best model with updated colors
    fig.add_trace(go.Bar(
        x=categories,
        y=baseline,
        name='Baseline Score',
        marker_color='rgb(158,202,225)',
        text=baseline,
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=best_model,
        name='XGBoost Score',
        marker_color='rgb(8,48,107)',
        text=best_model,
        textposition='auto'
    ))

    # Add line for comparison to baseline
    fig.add_trace(go.Scatter(
        x=categories,
        y=comparison_to_baseline,
        name='Model VS. Baseline',
        mode='lines+markers',
        marker=dict(color='rgb(255,99,71)', size=10),
        line=dict(color='rgb(255,99,71)', width=2),
        yaxis='y2'
    ))

    # Update layout for secondary y-axis
    fig.update_layout(
        title=dict(
            text='Performance Comparison to Baseline',
            x=0.5,
            xanchor='center',
            font=dict(size=23)
        ),
        xaxis_title='Case Outcome Classes',
        yaxis_title='Performance (%)',
        yaxis2=dict(
            title='Comparison to Baseline (x)',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h',
            bgcolor='rgba(255,255,255,0.5)'
        ),
        barmode='group'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    ### plot explained
    st.markdown("""
    ### Training Results Explained
                
    The above plot illustrates the trained model's performance in predicting the outcome of cases in the test set.
                
    The 'Baseline Score-to-Beat' represents the percentage of cases in the test set that belong to a particular case outcome.            
    The reason for establishing this score is so we know what the absolute minimum accuracy the model needs to achieve to be considered better than randomly guessing what the case outcome could be.
                
    The 'Model Score' represents the accuracy of the model in predicting the outcome of the cases.
    The 'Comparison to Baseline' line shows the model's learned capacity to predict a case outcome compared to the baseline.
    The higher the 'Comparison to Baseline' score, the better the model is at predicting that particular class.
                
    For example, if the 'Baseline Score-to-Beat' for 'Full Eviction' is 61.68%, it means that 61.68% of the cases in the test set resulted in 'Full Eviction'.
    Conversely, if the 'Model Score' for 'Full Eviction' is 88.34%, it means that the model was able to predict 'Full Eviction' with an accuracy of 88.34%.
    As the model score is higher than the baseline, (the 'Comparison to Baseline' being > 100%), this indicates that the model is ~1.4x better at predicting 'Full Eviction' than randomly guessing.
                
    To see how well the model learned the other case outcomes in comparison their respective baseline scores, refer to the orange line in the plot.

    """.replace("    ", "").strip()
    )

    st.markdown("""
    After training and validating the model on the training and validation sets, respectively, the model was then tested on a separate test set of cases to evaluate its performance in predicting the outcome of cases it had not seen before.
    The test set was comprised of a representatively distributed subset of cases from the original dataset, with the distribution, below:

    **Test Set Distribution (5280 cases):**
                
    - Full Eviction: 0.6169 (61.69%)
    - Dismissal: 0.1263 (12.63%)
    - Conditional Eviction: 0.0979 (9.79%)
    - No Eviction, Payment Plan: 0.0854 (8.54%)
    - Abatement: 0.0437 (4.37%)
    - Relief From Eviction: 0.0195 (1.95%)
    - Rent Adjustment: 0.0095 (0.95%)
    - Postponement: 0.0008 (0.08%)

    The trained model's performance in predicting the outcome of the cases in the test set is as follows:

    | Class (Case Outcome)      | Precision | Recall | F1-Score | Support (absolute) | Support (proportion) |
    |---------------------------|-----------|--------|----------|--------------------|----------------------|
    | Full Eviction             | 0.87      | 0.93   | **0.90**     | 3257      |  **0.6169**   |
    | Dismissal                 | 0.67      | 0.79   | **0.72**     | 667       |  **0.1263**   |
    | Conditional Eviction      | 0.67      | 0.52   | **0.59**     | 517       |  **0.0979**   |
    | No Eviction, Payment Plan | 0.68      | 0.52   | **0.59**     | 451       |  **0.0854**   |
    | Abatement                 | 0.72      | 0.62   | **0.67**     | 231       |  **0.0437**   |
    | Relief From Eviction      | 0.48      | 0.19   | **0.28**     | 103       |  **0.0195**   |
    | Rent Adjustment           | 0.88      | 0.42   | **0.57**     | 50        |  **0.0095**   |
    | Postponement              | 1.00      | 0.50   | **0.67**     | 4         |  **0.0008**   |
    
    | Metric                   | Value | Support |
    |--------------------------|-------|---------|
    | Accuracy                 | 0.80  | 5280    |
    | Macro Avg F1-Score       | 0.62  | 5280    |
    | Weighted Avg F1-Score    | 0.79  | 5280    |
                
    *For more information on the definitions of 'accuracy', 'precision', 'recall', and 'f1-score', please refer to the [this article](https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd).
                'Support' refers to the number of cases in the test set that relate to a given class (case outcome).*
    """.replace("    ", "").strip())

    st.divider()

    ### Model training mean fit time and mean score per iteration
    # Manually input data from the CSV file
    data = [
        {
            'model': 'RandomForest',
            'best_params': {
                "classifier__max_depth": 98,
                "classifier__max_features": 0.38620312747467755,
                "classifier__min_samples_leaf": 1,
                "classifier__min_samples_split": 2,
                "classifier__n_estimators": 426
            },
            'val_score': 0.8051346801346801,
            'training_iterations': 100,
            'mean_fit_time': [
                50.236834605534874, 25.90415628751119, 88.7272499402364, 36.90696374575297,
                4.49505074818929, 6.392673333485921, 12.494426170984903, 18.458106597264607,
                25.12930202484131, 19.76669955253601, 76.6315864721934, 37.47466778755188,
                12.689375162124634, 70.12734897931416, 15.105297644933065, 40.620591481526695,
                7.0448479652404785, 18.607242743174236, 80.83262228965759, 95.51320171356201,
                10.808812300364176, 70.0422150293986, 81.24414014816284, 11.253838777542114,
                16.929080724716187, 102.53905932108562, 57.67023491859436, 28.30555574099223,
                41.357780853907265, 17.915025393168133, 69.51184542973836, 46.21720941861471,
                13.05539059638977, 69.89430483182271, 54.528894344965614, 121.44622310002644,
                12.506099621454874, 30.934245665868122, 60.558074712753296, 58.180627981821694,
                64.04495771725972, 111.45345997810364, 33.542928298314415, 112.67973899841309,
                31.474718809127808, 25.973956028620403, 99.42514864603679, 14.058164596557617,
                9.706870317459106, 31.500919103622437, 100.06755685806274, 51.05634832382202,
                38.24394941329956, 35.76431608200073, 43.86011242866516, 47.183152755101524,
                65.73028167088826, 35.12653136253357, 111.81853850682576, 143.62991881370544,
                9.826116641362509, 66.5938971042633, 47.09293421109518, 76.7668293317159,
                86.00674104690552, 21.64133636156718, 71.99862066904704, 37.65360458691915,
                64.04215304056804, 62.67858386039734, 85.38796257972717, 50.353399912516274,
                8.14829675356547, 34.825411319732666, 136.52191837628683, 63.03057066599528,
                22.146161317825317, 18.266755263010662, 50.039453983306885, 45.9700243473053,
                44.649099349975586, 66.94249192873637, 16.286324977874756, 19.526768287022907,
                36.413638989130654, 51.222079277038574, 91.57798274358113, 11.767161051432291,
                27.170483430226643, 67.0619797706604, 38.97104072570801, 62.471253395080566,
                8.253347873687744, 8.544302543004354, 40.16099691390991, 136.57484674453735,
                30.956113656361897, 43.494224309921265, 90.65773606300354, 58.226386308670044
            ],
            'mean_score_time': [
                0.09710574150085449, 0.5779253641764323, 0.7403313318888346, 0.6667975584665934,
                0.0785214106241862, 0.07695865631103516, 0.056998491287231445, 0.7133771578470866,
                0.9099492232004801, 0.06973656018575032, 0.8537602424621582, 0.3740796248118083,
                0.4149339199066162, 0.3819444179534912, 0.10233434041341145, 0.5691210428873698,
                0.11932094891866048, 0.7807356516520182, 0.5204619566599528, 0.5045749346415201,
                0.1166676680246989, 0.5491722424825033, 0.6284758249918619, 0.06012153625488281,
                0.5195803642272949, 0.9625459512074789, 0.30641595522562665, 0.3902595043182373,
                0.5598057905832926, 0.24264327685038248, 1.055029312769572, 0.5206092993418375,
                0.5105883280436198, 0.7135906219482422, 0.534272034962972, 0.8082387447357178,
                0.4012076059977214, 1.2621334393819172, 0.6573127110799154, 0.6405660311381022,
                0.48099271456400555, 0.720430056254069, 0.139966090520223, 0.9427235921223959,
                0.2399007479349772, 0.3602856794993083, 0.5485599835713705, 0.05474098523457845,
                0.14038705825805664, 0.35942959785461426, 0.674854040145874, 0.11145774523417155,
                0.22415669759114584, 0.7499156792958578, 0.3427245616912842, 0.8187149365743002,
                0.23123065630594888, 0.2352899710337321, 0.5263023376464844, 0.8909463087717692,
                0.07152080535888672, 0.5150482654571533, 0.14191937446594238, 0.38726162910461426,
                0.54121994972229, 0.40188026428222656, 0.5670503775278727, 0.16948374112447104,
                0.9548802375793457, 0.32976484298706055, 0.5212926069895426, 0.3960440158843994,
                0.09490362803141277, 0.2902393341064453, 1.0842790603637695, 0.32094573974609375,
                0.6491492589314779, 0.2575539747873942, 0.23630595207214355, 0.16323169072469076,
                0.1445030371348063, 0.45446228981018066, 0.1693437099456787, 0.14119402567545572,
                0.14782937367757162, 0.5517379442850748, 0.5192579428354899, 0.0538936456044515,
                0.8932653268178304, 0.2073535919189453, 0.14006304740905762, 0.7031175295511881,
                0.07904672622680664, 0.15374072392781576, 0.5061683654785156, 0.8323630491892496,
                0.14794166882832846, 0.5490287939707438, 0.39987866083780926, 0.1361093521118164
            ],
            'mean_val_score': [
                0.7167912693680615, 0.7082786910494233, 0.7422590749982231, 0.7373245348751499,
                0.6661366535352581, 0.7296538484757239, 0.7397099401593916, 0.7209306667396724,
                0.7137979181717649, 0.7169316579736317, 0.7613423632124486, 0.6819456689582766,
                0.7368802102298572, 0.7184517040538431, 0.7402010894457686, 0.7118801538709651,
                0.6686623571852138, 0.7093077141790568, 0.7413937715788314, 0.7117631882506473,
                0.7094246141703883, 0.7644059685979805, 0.7778999231658487, 0.7109680996430988,
                0.7408324878761195, 0.7527361470967149, 0.7370439365029974, 0.7307997256563983,
                0.7400139713611446, 0.7390785154774463, 0.7854069243102507, 0.7371842381501607,
                0.6498363083461501, 0.71894285334022, 0.713774444324048, 0.7257482855419938,
                0.7839569601431852, 0.7836996928757042, 0.6806593703575388, 0.7763329900970158,
                0.7343077098311367, 0.7242749180783715, 0.7001869766541583, 0.7479419397945732,
                0.7612020221878936, 0.7095883174733834, 0.7569924346153094, 0.7093077010532595,
                0.6536014597041649, 0.7260523249473697, 0.6933581460780021, 0.7496959154746957,
                0.7144292706618254, 0.7738307570133277, 0.7134236327807769, 0.7247660329095305,
                0.7629092011192511, 0.7284844925251587, 0.7535545979827033, 0.7487838448395828,
                0.7602431908999582, 0.7568989329983603, 0.7399671270313645, 0.7489709120617424,
                0.7415340863517917, 0.7061037513617235, 0.7192000894339324, 0.7256781166704408,
                0.7688494037342428, 0.7221233898908516, 0.7123478522797703, 0.7373479529382282,
                0.7242749377670675, 0.7224040079516998, 0.7332085309557473, 0.7333956145851536,
                0.7014265096798997, 0.7507249632151992, 0.7623479485903079, 0.7120438358445397,
                0.7366930724565371, 0.6590970948466142, 0.7612254139993775, 0.7275724071235238,
                0.7418848667212942, 0.7400373779391503, 0.744527521225046, 0.77369037989283,
                0.7125817982869279, 0.716534018507827, 0.7588633611492276, 0.7598222580661494,
                0.7295134959660964, 0.7041393346959289, 0.7228249752407265, 0.7520345207287233,
                0.7333488326029105, 0.7254442461366176, 0.7133300753791896, 0.7302384239057154
            ]
            },
        {
            'model': 'XGBoost',
            'best_params': {
                "classifier__colsample_bytree": 0.625125680257932,
                "classifier__gamma": 0.194173672147116,
                "classifier__learning_rate": 0.10097965440196684,
                "classifier__max_depth": 13,
                "classifier__min_child_weight": 5,
                "classifier__n_estimators": 313,
                "classifier__subsample": 0.8849967765493054
            }, 
            'val_score': 0.8080808080808081,
            'training_iterations': 100,
            'mean_fit_time': [
                5.408519983291626, 6.363030751546224, 11.731470982233683, 16.32541831334432,
                13.289165019989014, 14.350188334782919, 23.33161536852519, 10.308293660481771,
                11.821661233901978, 15.760297298431396, 15.126868089040121, 23.02529803911845,
                18.6081600189209, 12.470905621846518, 2.892112414042155, 4.794827302296956,
                8.92768398920695, 13.484126488367716, 19.081993023554485, 16.76933677991231,
                16.76311167081197, 13.232101202011108, 16.423155625661213, 9.542754411697388,
                8.32668137550354, 15.075334390004477, 5.916849295298259, 12.69831625620524,
                6.872479677200317, 13.529250383377075, 4.131145079930623, 2.7596680323282876,
                6.740423679351807, 8.477234999338785, 17.55939292907715, 29.14831797281901,
                16.555116573969524, 10.906092802683512, 4.149724721908569, 4.970834970474243,
                7.112377564112346, 8.580732742945353, 34.620728731155396, 8.213569243748983,
                14.256858428319296, 15.357655048370361, 19.551368872324627, 22.9227458635966,
                16.17630871136983, 5.703083435694377, 10.968394676844278, 4.575952053070068,
                13.429558753967285, 4.648947397867839, 4.100438356399536, 8.743781646092733,
                12.151322285334269, 6.462501049041748, 14.157079776128134, 11.656432310740152,
                15.883915424346924, 8.520919879277548, 7.4496387640635175, 8.77682344118754,
                20.348410765329998, 15.740500688552856, 5.7468531131744385, 5.814268747965495,
                11.715343634287516, 11.02298871676127, 4.7148440678914385, 6.342104276021321,
                8.06408723195394, 15.864017645517984, 16.480456988016766, 13.839460770289103,
                11.769019683202108, 2.9804739157358804, 8.946569760640463, 4.688107252120972,
                16.718225320180256, 12.43903390566508, 10.050039688746134, 11.960329929987589,
                14.364361047744751, 3.387007236480713, 8.915050427118937, 3.7330539226531982,
                4.449030001958211, 11.770737727483114, 10.67691437403361, 7.291922728220622,
                13.69331169128418, 5.188038905461629, 9.052915732065836, 23.29212236404419,
                13.906167586644491, 12.05799126625061, 10.446734110514322, 10.489616632461548,
            ],
            'mean_score_time': [
                0.18304705619812012, 0.22925527890523276, 1.002594232559204, 0.6927820841471354,
                0.9749236106872559, 0.5166786511739095, 1.3566816647847493, 0.5981606642405192,
                0.515138308207194, 0.6215193271636963, 0.6082780361175537, 1.4559190273284912,
                1.5578029950459797, 1.1747926870981853, 0.18178772926330566, 0.28519614537556964,
                0.3244446913401286, 1.1709943612416585, 1.4309895038604736, 0.7041064103444418,
                0.8234156767527262, 1.2026867071787517, 0.3860170046488444, 0.33836905161539715,
                0.3468799591064453, 1.0531892776489258, 0.21367192268371582, 0.4218564033508301,
                0.31417202949523926, 0.3274887402852376, 0.20278255144755045, 0.17291855812072754,
                0.21888860066731772, 0.5785946051279703, 0.9424480597178141, 1.8621833324432373,
                0.703147808710734, 0.31256532669067383, 0.2547484238942464, 0.2243343989054362,
                0.36534349123636883, 0.318328857421875, 2.6857295831044516, 0.38714782396952313,
                0.7436042626698812, 1.4432623386383057, 1.7371280988057454, 2.089200019836426,
                0.936518669128418, 0.18801585833231607, 0.8229240576426188, 0.23237061500549316,
                0.453868309656779, 0.3473563989003499, 0.2848045825958252, 0.6706337134043375,
                0.33812443415323895, 0.20265396436055502, 0.8680590788523356, 0.5071189403533936,
                1.420023520787557, 0.3071328004201253, 0.4388793309529622, 0.38182806968688965,
                1.1991531054178874, 0.6032330195109049, 0.2515100638071696, 0.2956785360972087,
                0.6978015899658203, 0.5255920886993408, 0.4386881987253825, 0.24562366803487143,
                0.34560052553812665, 0.7571242650349935, 0.6375493208567301, 1.0579842726389568,
                0.2639368375142415, 0.20661664009094238, 0.7105007171630859, 0.3616789976755778,
                1.2840009530385335, 0.5555950800577799, 0.3130307197570801, 0.3893094062805176,
                0.6108503341674805, 0.1318670113881429, 0.3876403172810872, 0.23041431109110513,
                0.23294933636983237, 0.37330158551534015, 0.29718637466430664, 0.2815558910369873,
                0.580721934636434, 0.4215375582377116, 0.7261722882588705, 1.7271180152893066,
                0.3285371462504069, 0.8655431270599365, 0.4335480531056722, 0.3096551100413005
            ],
            'mean_val_score': [
                0.757390064236766, 0.7592843432047762, 0.7545837113521933, 0.7699718973070567,
                0.7746024111505512, 0.765832522905591, 0.7801917677670538, 0.7639382439375807,
                0.7664639475875363, 0.764803506338856, 0.762043889496236, 0.7574601904494775,
                0.7856875571376211, 0.7814780105831532, 0.757390029781548, 0.7657857327197246,
                0.763119725781881, 0.7737137306861973, 0.7739241765940434, 0.7632366274139372,
                0.7684284216786942, 0.7809167268804414, 0.7659026770106219, 0.7544433736090875,
                0.7552151278305154, 0.7654583622096771, 0.7529232356482951, 0.7592375809112291,
                0.7510990960187938, 0.7576940478575036, 0.762277789563103, 0.7572731133829699,
                0.7591440431983377, 0.7710243335775946, 0.7645228800743844, 0.7647567702969035,
                0.7745089144557763, 0.7605004335565693, 0.7557296213473609, 0.7573198330176759,
                0.767492960872822, 0.7545836637711781, 0.7844247323846001, 0.7685453512030694,
                0.7688727561683347, 0.7890318216021175, 0.7646865915810027, 0.7914639891799066,
                0.7612955599007852, 0.7568287575639089, 0.7572263346821764, 0.7508184795986702,
                0.7624414977882719, 0.7672357198569354, 0.7616931665520966, 0.7565013706466148,
                0.7562207230527226, 0.7586763054121407, 0.7813611352026917, 0.7592843645341968,
                0.7893592101601362, 0.7626051699174985, 0.7718896452006098, 0.7610149188697916,
                0.7764031868608879, 0.7709073794423493, 0.7621375026824618, 0.7708138220407621,
                0.7537651849928704, 0.7773620099452, 0.7753273817490114, 0.762394701039507,
                0.7592375694261565, 0.768592115137341, 0.7632133077943383, 0.7552151311119647,
                0.7600794777526153, 0.7634471143398999, 0.7622310371139038, 0.7702057875295757,
                0.7777362215035781, 0.7737137241232986, 0.7594480317412492, 0.7593311235462945,
                0.74773148076093, 0.7568522314116257, 0.7610616746004398, 0.761973784612945,
                0.7628390880323367, 0.7615528140424689, 0.7565481247365385, 0.7644761177808373,
                0.7623011633266156, 0.7646164555239431, 0.776753983637637, 0.7796305037530379,
                0.7554255638940136, 0.7625350486269608, 0.7694106250894174, 0.7636809947180708
            ]
        },
    ]

    ### Mean Validation Time Plot
    # Add lines for each model's mean fit time
    fig0 = go.Figure()
    for row in data:
        training_iterations = list(range(row['training_iterations']))
        mean_fit_time = row['mean_fit_time']
        
        fig0.add_trace(go.Scatter(
            x=training_iterations,
            y=mean_fit_time,
            mode='lines',
            name=row['model'],
            line=dict(width=2)
        ))

    # Update layout for the first plot
    fig0.update_layout(
        title=dict(
            text='Training Results: Mean Fit Time',
            x=0.5,
            xanchor='center',
            font=dict(size=23)
        ),
        xaxis_title='Training Iterations',
        yaxis_title='Mean Fit Time (s)',
        xaxis=dict(
            tickvals=[i for i in range(0, max(training_iterations), 10)] + [max(training_iterations)]
        ),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h',
            bgcolor='rgba(255,255,255,0.5)'
        ),
        showlegend=True
    )

    st.plotly_chart(fig0)

    ### Mean Validation Score Plot
    # Add lines for each model's mean test score
    fig1 = go.Figure()
    for row in data:
        training_iterations = list(range(row['training_iterations']))
        mean_val_score = row['mean_val_score']
        
        fig1.add_trace(go.Scatter(
            x=training_iterations,
            y=mean_val_score,
            mode='lines',
            name=row['model'],
            line=dict(width=2)
        ))

    # Update layout for the first plot
    fig1.update_layout(
        title=dict(
            text='Training Results: Mean Validation Score',
            x=0.5,
            xanchor='center',
            font=dict(size=23)
        ),
        xaxis_title='Training Iterations',
        yaxis_title='Mean Validation Score (%)',
        xaxis=dict(
            tickvals=[i for i in range(0, max(training_iterations), 10)] + [max(training_iterations)]
        ),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h',
            bgcolor='rgba(255,255,255,0.5)'
        ),
        showlegend=True
    )

    st.plotly_chart(fig1)


    st.markdown("""
    ### Training Process
                
    The above plot illustrates the mean validation score of the model during the training process.
                
    The 'Training Iterations' represent the number of times the model was trained on the dataset.
    The 'Mean Validation Score' represents the average performance of the model on the validation set after each training iteration.
                
    The plot shows how the model's performance improved over time as it learned from the data.
    The goal of the training process is to maximize the mean validation score, which indicates that the model is making accurate predictions on unseen data.
                
    The plot allows us to see how the model's performance changed over time and how well it learned from the data.
                
    The training process involves adjusting the model's parameters to minimize the error between the predicted and actual outcomes.
    By training the model multiple times on the dataset, it can learn the underlying patterns and relationships in the data to make accurate predictions.
    """.replace("    ", "").strip())

# Function to show the data exploration page
def show_data_exploration():
    """Function to show the data exploration page."""
    st.markdown("## Explore the Data")

    # st.divider()

    data_explore_wr = """
    On this page, you can explore some of the data extracted from the case files used to train the model in this project.
    The data includes information about various aspects of the cases, such as the landlord's representation, the tenant's financial situation, and the history of late payments.
    To respect the anonymity and privacy of the individuals involved in the cases, the data has been anonymized and aggregated to provide a high-level overview of the trends and patterns observed in the dataset.

    Using the **sidebar** *(accessible via the arrow in the top left-hand corner of this page)*, select one or more columns to generate simple graphs using their data.
    With each column, there is also a definiton of the information in that column to explain how the data was collected from each case and what it represents.
    
    The graphs can help to understand the distribution of the data and identify any patterns or trends that may be present.
    You may even notice some interesting trends that the model has picked up in its learning.
    
    **Note: To keep the graphs manageable, for any columns with more than 10 unique values, only the top 10 highest counts are displayed.**
    """.replace("    ", "").strip()

    st.markdown(data_explore_wr)

    st.divider()

    # Load the JSON file with data distributions
    with open("./data/app-data/non-applications-distributions.json", "r") as f:
        data_distributions = json.load(f)

        # sort by "column" key in alphabetical order
        data_distributions = sorted(data_distributions, key=lambda x: x["column"])

    # Function to format column names for display
    def format_column_name(column_name):
        """Function to format column names for display."""
        return column_name.replace('_', ' ').title()

    # Function to plot categorical data
    def plot_categorical(column_name, distribution):
        """Function to plot categorical data."""
        labels, values = zip(*distribution.items())
        fig = go.Figure(data=[go.Bar(x=labels, y=values)])
        fig.update_layout(
            title=dict(text=f"Distribution of {format_column_name(column_name)}", x=0.4),
            xaxis_title=f"'{format_column_name(column_name)}' Values",
            yaxis_title="Count"
        )
        st.plotly_chart(fig)

    # Function to plot numerical data
    def plot_numerical(column_name, distribution):
        """Function to plot numerical data."""
        labels, values = zip(*distribution.items())
        fig = go.Figure(data=[go.Scatter(x=labels, y=values, mode='markers')])
        fig.update_layout(
            title=dict(text=f"Distribution of {format_column_name(column_name)}", x=0.4),
            xaxis_title=f"'{format_column_name(column_name)}' Values",
            yaxis_title="Count"
        )
        st.plotly_chart(fig)

    # Function to plot special values data
    def plot_special_values(column_name, distribution):
        labels, values = zip(*distribution.items())
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(
            title=dict(text=f"Distribution of {format_column_name(column_name)}", x=0.4)
        )
        st.plotly_chart(fig)


    # check that current tab is the "Explore the Data" tab
    # if st.session_state.current_tab == "Explore the Data":

    # Sidebar for selecting the column to plot
    st.sidebar.markdown("## Select Column to Plot")
    columns = [info["column"] for info in data_distributions]
    columns_display = [format_column_name(col) for col in columns]

    selected_columns_display = st.sidebar.multiselect("Columns", columns_display)

    if selected_columns_display:
        st.markdown(f"## Graphs")
        st.divider()

        for selected_column_display in selected_columns_display:
            selected_column = columns[columns_display.index(selected_column_display)]
            for info in data_distributions:
                if info["column"] == selected_column:
                    distribution = info["distribution"]
                    definition = info.get("definition", "No definition available")
                    break

            st.markdown(f"""
            **Column Name:** *{selected_column_display}*
            
            **Definition:** *{definition}*
            """.replace("    ", "").strip())

            # Determine the type of plot based on the data distribution
            if all(isinstance(v, int) for v in distribution.values()):
                plot_categorical(selected_column, distribution)
            elif any(key in cat_common_order_1 for key in distribution.keys()):
                if all(key in cat_common_order_1 for key in distribution.keys()):
                    plot_special_values(selected_column, distribution)
                else:
                    plot_numerical(selected_column, distribution)

            st.divider()

def main():
    """Main function for the Streamlit app."""
    tabs_labels = ["About", "Form", "Form Results", "Explore the Data", "Model Training"]
    st.sidebar.title("Navigation")
    selected_tab = st.sidebar.radio("Go to", tabs_labels)
    
    if selected_tab == "About":
        show_about()
    elif selected_tab == "Form":
        show_form()
    elif selected_tab == "Form Results":
        show_results()
    elif selected_tab == "Explore the Data":
        show_data_exploration()
    elif selected_tab == "Model Training":
        show_model_training()

if __name__ == "__main__":
    main()
