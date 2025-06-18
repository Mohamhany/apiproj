% Knowledge base of symptoms for each disease
has_symptom(common_cold, sneezing).
has_symptom(common_cold, runny_nose).
has_symptom(common_cold, sore_throat).
has_symptom(common_cold, mild_cough).

has_symptom(influenza, fever).
has_symptom(influenza, muscle_aches).
has_symptom(influenza, dry_cough).
has_symptom(influenza, fatigue).
has_symptom(influenza, sore_throat).

has_symptom(alcohol_poisoning, vomiting).
has_symptom(alcohol_poisoning, confusion).
has_symptom(alcohol_poisoning, slow_heartbeat).
has_symptom(alcohol_poisoning, unconsciousness).

has_symptom(otitis_media, ear_pain).
has_symptom(otitis_media, fever).
has_symptom(otitis_media, trouble_hearing).
has_symptom(otitis_media, irritability).

has_symptom(tonsillitis, sore_throat).
has_symptom(tonsillitis, swollen_tonsils).
has_symptom(tonsillitis, fever).
has_symptom(tonsillitis, difficulty_swallowing).

has_symptom(covid19, fever).
has_symptom(covid19, dry_cough).
has_symptom(covid19, loss_of_taste).
has_symptom(covid19, loss_of_smell).
has_symptom(covid19, shortness_of_breath).

has_symptom(strep_throat, sore_throat).
has_symptom(strep_throat, swollen_lymph_nodes).
has_symptom(strep_throat, fever).
has_symptom(strep_throat, headache).

has_symptom(pneumonia, chest_pain).
has_symptom(pneumonia, productive_cough).
has_symptom(pneumonia, fever).
has_symptom(pneumonia, chills).
has_symptom(pneumonia, shortness_of_breath).

has_symptom(sinusitis, facial_pain).
has_symptom(sinusitis, nasal_congestion).
has_symptom(sinusitis, thick_nasal_discharge).
has_symptom(sinusitis, headache).

has_symptom(bronchitis, persistent_cough).
has_symptom(bronchitis, mucus_production).
has_symptom(bronchitis, chest_discomfort).
has_symptom(bronchitis, fatigue).

has_symptom(malaria, fever).
has_symptom(malaria, chills).
has_symptom(malaria, sweating).
has_symptom(malaria, headache).
has_symptom(malaria, nausea).

has_symptom(food_poisoning, nausea).
has_symptom(food_poisoning, vomiting).
has_symptom(food_poisoning, diarrhea).

has_symptom(asthma, wheezing).
has_symptom(asthma, shortness_of_breath).
has_symptom(asthma, chest_tightness).
has_symptom(asthma, coughing_at_night).

has_symptom(chickenpox, itchy_rash).
has_symptom(chickenpox, fever).
has_symptom(chickenpox, fatigue).
has_symptom(chickenpox, loss_of_appetite).

has_symptom(drug_overdose, confusion).
has_symptom(drug_overdose, slow_breathing).
has_symptom(drug_overdose, nausea).
has_symptom(drug_overdose, unconsciousness).

has_symptom(urinary_tract_infection, burning_urination).
has_symptom(urinary_tract_infection, frequent_urination).
has_symptom(urinary_tract_infection, cloudy_urine).
has_symptom(urinary_tract_infection, pelvic_pain).

has_symptom(dengue_fever, high_fever).
has_symptom(dengue_fever, rash).
has_symptom(dengue_fever, joint_pain).
has_symptom(dengue_fever, headache).

has_symptom(conjunctivitis, red_eyes).
has_symptom(conjunctivitis, eye_discharge).
has_symptom(conjunctivitis, itching_eyes).
has_symptom(conjunctivitis, watery_eyes).

% Treatments for each disease
treatment(common_cold, 'Rest, hydration, and over-the-counter remedies.').
treatment(influenza, 'Antivirals if early, rest, and fluids.').
treatment(covid19, 'Isolation, hydration, monitoring, and medical attention if severe.').
treatment(strep_throat, 'Antibiotics, pain relievers, and rest.').
treatment(pneumonia, 'Antibiotics (if bacterial), rest, fluids, and oxygen if needed.').
treatment(sinusitis, 'Nasal decongestants, saline rinses, and possibly antibiotics.').
treatment(bronchitis, 'Rest, fluids, cough suppressants, and bronchodilators if needed.').
treatment(malaria, 'Visit a doctor immediately. Antimalarial drugs are necessary.').
treatment(food_poisoning, 'Stay hydrated, rest, and eat bland food. See a doctor if symptoms are severe.').
treatment(otitis_media, 'Pain relievers, warm compress, and antibiotics if bacterial.').

treatment(tonsillitis, 'Rest, warm fluids, pain relievers, and antibiotics if bacterial.').

treatment(asthma, 'Inhaled bronchodilators, corticosteroids, and avoiding triggers.').
treatment(chickenpox, 'Antihistamines, calamine lotion, rest, and fluids.').
treatment(hepatitis_a, 'Supportive care, rest, hydration, and avoiding alcohol.').
treatment(drug_overdose, 'Emergency medical care, activated charcoal, and antidotes if available.').
treatment(urinary_tract_infection, 'Antibiotics, hydration, and urinary alkalinizers.').
treatment(alcohol_poisoning, 'Emergency medical care, oxygen therapy, IV fluids, and monitoring.').
treatment(dengue_fever, 'Rest, fluids, and pain relievers like acetaminophen. Avoid aspirin.').
treatment(conjunctivitis, 'Warm compresses, artificial tears, and antibiotics if bacterial.').

% Check if a disease has all the given symptoms
has_all_symptoms(_, []).
has_all_symptoms(Disease, [H|T]) :-
    has_symptom(Disease, H),
    has_all_symptoms(Disease, T).

% Main diagnosis predicate (returning one match at a time)
diagnose(Symptoms, Disease, Treatment) :-
    has_all_symptoms(Disease, Symptoms),
    treatment(Disease, Treatment).
