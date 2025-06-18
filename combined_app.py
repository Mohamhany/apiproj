from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array 
from PIL import Image
import io
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyswip import Prolog
import traceback
import gc
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

datagen = ImageDataGenerator()

# --- Brain Tumor Model Loading and Prediction ---

brain_model = None
try:
    brain_model = load_model('Brain tumor classification_edit.h5')
    print("Brain Tumor Model loaded successfully!")
except Exception as e:
    print(f"Error loading Brain Tumor model: {str(e)}")

brain_class_names = ['brain_glioma', 'brain_menin', 'brain_tumor']

def preprocess_brain_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = datagen.flow(img_array, batch_size=1).__next__()
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing brain image: {str(e)}")

@app.route('/brain/predict', methods=['POST'])
def predict_brain_tumor():
    if brain_model is None:
        return jsonify({'error': 'Brain Tumor Model not loaded properly'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        
        try:
            Image.open(io.BytesIO(img_bytes))
        except:
            return jsonify({'error': 'Invalid image file'}), 400

        input_data = preprocess_brain_image(img_bytes)
        
        prediction = brain_model.predict(input_data)[0]
        predicted_class = int(np.argmax(prediction))

        class_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(brain_class_names, prediction)
        }

        return jsonify({
            'predicted_class_name': brain_class_names[predicted_class],
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# --- Lymphoma Model Loading and Prediction ---

lymphoma_model = None
try:
    lymphoma_model = load_model('Lymphoma classification1.h5')
    print("Lymphoma Model loaded successfully!")
except Exception as e:
    print(f"Error loading Lymphoma model: {str(e)}")

lymphoma_class_names = ['lymph_cll', 'lymph_fl','lymph_mcl']

def preprocess_lymphoma_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0);
        img_array=datagen.flow(img_array,batch_size=1).__next__()
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing lymphoma image: {str(e)}")

@app.route('/lymphoma/predict', methods=['POST'])
def predict_lymphoma():
    if lymphoma_model is None:
        return jsonify({'error': 'Lymphoma Model not loaded properly'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        
        try:
            Image.open(io.BytesIO(img_bytes))
        except:
            return jsonify({'error': 'Invalid image file'}), 400

        input_data = preprocess_lymphoma_image(img_bytes)
        
        prediction = lymphoma_model.predict(input_data)[0]
        predicted_class = int(np.argmax(prediction))

        class_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(lymphoma_class_names, prediction)
        }

        return jsonify({
            'predicted_class_name': lymphoma_class_names[predicted_class],
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# --- Acute Lymphoblastic Leukemia Model Loading and Prediction ---

lymphoblastic_model = None
try:
    lymphoblastic_model = load_model('Acute Lymphoblastic Leukemia classification.h5')
    print("Acute Lymphoblastic Leukemia Model loaded successfully!")
except Exception as e:
    print(f"Error loading Acute Lymphoblastic Leukemia model: {str(e)}")

lymphoblastic_class_names = ['all_benign', 'all_early','all_pre','all_pro']

def preprocess_lymphoblastic_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0);
        img_array=datagen.flow(img_array,batch_size=1).__next__()
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing lymphoblastic image: {str(e)}")

@app.route('/lymphoblastic/predict', methods=['POST'])
def predict_lymphoblastic():
    if lymphoblastic_model is None:
        return jsonify({'error': 'Acute Lymphoblastic Leukemia Model not loaded properly'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        
        try:
            Image.open(io.BytesIO(img_bytes))
        except:
            return jsonify({'error': 'Invalid image file'}), 400

        input_data = preprocess_lymphoblastic_image(img_bytes);
        
        prediction = lymphoblastic_model.predict(input_data)[0]
        predicted_class = int(np.argmax(prediction))

        class_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(lymphoblastic_class_names, prediction)
        }

        return jsonify({
            'predicted_class_name': lymphoblastic_class_names[predicted_class],
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# --- Teeth Model Loading and Prediction ---

teeth_model = None
try:
    teeth_model = load_model('teeth_classification_pretrain (1).h5');
    print("Teeth Model loaded successfully!");
except Exception as e:
    print(f"Error loading Teeth model: {str(e)}")

teeth_class_names = ['CaS', 'CoS','Gum','MC','OC','OLP','OT'];

def preprocess_teeth_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB');
        image = image.resize((224, 224));
        img_array = img_to_array(image);
        img_array = np.expand_dims(img_array, axis=0);
        img_array=datagen.flow(img_array,batch_size=1).__next__();
        return img_array;
    except Exception as e:
        raise Exception(f"Error preprocessing teeth image: {str(e)}")

@app.route('/teeth/predict', methods=['POST'])
def predict_teeth():
    if teeth_model is None:
        return jsonify({'error': 'Teeth Model not loaded properly'}), 500;

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400;

    try:
        file = request.files['image'];
        img_bytes = file.read();
        
        try:
            Image.open(io.BytesIO(img_bytes));
        except:
            return jsonify({'error': 'Invalid image file'}), 400;

        input_data = preprocess_teeth_image(img_bytes);
        
        prediction = teeth_model.predict(input_data)[0];
        predicted_class = int(np.argmax(prediction));

        class_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(teeth_class_names, prediction)
        };

        return jsonify({
            'predicted_class_name': teeth_class_names[predicted_class],
            'status': 'success'
        });

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500;

# --- Breast Cancer Model Loading and Prediction ---

breast_model = None
try:
    breast_model = load_model('Breast Cancer.h5')
    print("Breast Cancer Model loaded successfully!")
except Exception as e:
    print(f"Error loading Breast Cancer model: {str(e)}")

breast_class_names = ['Benign', 'Malignant']

def preprocess_breast_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = datagen.flow(img_array, batch_size=1).__next__()
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing breast image: {str(e)}")

@app.route('/breast', methods=['POST'])
def predict_breast():
    if breast_model is None:
        return jsonify({'error': 'Breast Cancer Model not loaded properly'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        
        try:
            Image.open(io.BytesIO(img_bytes))
        except:
            return jsonify({'error': 'Invalid image file'}), 400

        input_data = preprocess_breast_image(img_bytes)
        
        prediction = breast_model.predict(input_data)[0]
        predicted_class = int(np.argmax(prediction))

        return jsonify({
            'predicted_class_name': breast_class_names[predicted_class],
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# --- Recommendation System Loading and Prediction ---

# Load and prepare data
try:
    df_recommend = pd.read_csv('Medicine_Details.csv');
    df_recommend.fillna('', inplace=True);
    df_recommend['combined_text'] = df_recommend['Uses'] + ' ' + df_recommend['Composition'] + ' ' + df_recommend['Side_effects'];

    vectorizer_recommend = TfidfVectorizer(stop_words='english');
    tfidf_matrix_recommend = vectorizer_recommend.fit_transform(df_recommend['combined_text']);
    print("Recommendation System data loaded successfully!");
except Exception as e:
    print(f"Error loading Recommendation System data: {str(e)}")
    df_recommend = None
    vectorizer_recommend = None
    tfidf_matrix_recommend = None

def recommend_medicine(user_input, top_n=5):
    if df_recommend is None or vectorizer_recommend is None or tfidf_matrix_recommend is None:
        return None;
    try:
        user_tfidf = vectorizer_recommend.transform([user_input]);
        cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix_recommend).flatten();
        top_indices = cosine_similarities.argsort()[-top_n:][::-1];
        return df_recommend.iloc[top_indices][['Medicine Name', 'Uses', 'Side_effects']].to_dict(orient='records');
    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
        return None;

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    if request.method == 'POST':
        data = request.get_json(); # Assuming input is JSON
        query = data.get('query', None);

        if query is None:
            return jsonify({'error': 'Missing query parameter'}), 400;

        recommendations = recommend_medicine(query);

        if recommendations is None:
             return jsonify({'error': 'Error generating recommendations'}), 500; # Handle potential errors in recommendation function

        return jsonify({'recommendations': recommendations});
    

# --- Cervical Cancer Model Loading and Prediction ---

cervical_model = None
try:
    cervical_model = load_model('Cervical Cancer.h5')
    print("Cervical Cancer Model loaded successfully!")
except Exception as e:
    print(f"Error loading Cervical Cancer model: {str(e)}")

cervical_class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']

def preprocess_cervical_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = datagen.flow(img_array, batch_size=1).__next__()
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing cervical image: {str(e)}")

@app.route('/cervical', methods=['POST'])
def predict_cervical():
    if cervical_model is None:
        return jsonify({'error': 'Cervical Cancer Model not loaded properly'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        
        try:
            Image.open(io.BytesIO(img_bytes))
        except:
            return jsonify({'error': 'Invalid image file'}), 400

        input_data = preprocess_cervical_image(img_bytes)
        
        prediction = cervical_model.predict(input_data)[0]
        predicted_class = int(np.argmax(prediction))

        return jsonify({
            'predicted_class_name': cervical_class_names[predicted_class],
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# --- Kidney Model Loading and Prediction ---

kidney_model = None
try:
    # Register the custom activation function
    get_custom_objects().update({'softmax_v2': tf.keras.activations.softmax})
    
    kidney_model = load_model('ct_kidney.h5', custom_objects={'softmax_v2': tf.keras.activations.softmax})
    print("Kidney Model loaded successfully!")
except Exception as e:
    print(f"Error loading Kidney model: {str(e)}")

kidney_class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

def preprocess_kidney_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = datagen.flow(img_array, batch_size=1).__next__()
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing kidney image: {str(e)}")

@app.route('/kidney', methods=['POST'])
def predict_kidney():
    if kidney_model is None:
        return jsonify({'error': 'Kidney Model not loaded properly'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        
        try:
            Image.open(io.BytesIO(img_bytes))
        except:
            return jsonify({'error': 'Invalid image file'}), 400

        input_data = preprocess_kidney_image(img_bytes)
        
        prediction = kidney_model.predict(input_data)[0]
        predicted_class = int(np.argmax(prediction))

        return jsonify({
            'predicted_class_name': kidney_class_names[predicted_class],
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# --- Prolog Diagnosis Integration ---
prolog = Prolog()
try:
    prolog.consult("expert112123231.pl")
    print("Prolog file loaded successfully")
except Exception as e:
    print(f"Error loading Prolog file: {e}")

all_symptoms = [
    "sneezing", "runny_nose", "sore_throat", "mild_cough", "fever", "muscle_aches", "dry_cough",
    "fatigue", "loss_of_taste", "loss_of_smell", "shortness_of_breath", "swollen_lymph_nodes",
    "headache", "chest_pain", "productive_cough", "chills", "facial_pain", "nasal_congestion",
    "thick_nasal_discharge", "persistent_cough", "mucus_production", "chest_discomfort",
    "sweating", "nausea", "vomiting", "diarrhea"
]

@app.route("/diagnoses", methods=["POST"])
def diagnoses():
    data = request.get_json()
    selected_symptoms = data.get("symptoms", [])
    diagnosis_results = []
    error = None

    if selected_symptoms:
        try:
            # Prolog expects atoms, so wrap each symptom in single quotes
            symptom_list = "[" + ",".join([f"'{s}'" for s in selected_symptoms]) + "]"
            query = prolog.query(f"diagnose({symptom_list}, Disease, Treatment)", maxresult=10)
            for result in query:
                diagnosis_results.append({
                    "disease": str(result["Disease"]).replace('_', ' ').title(),
                    "treatment": str(result["Treatment"])
                })
            query.close()
        except Exception as e:
            error = f"Error querying Prolog: {e}"
    else:
        error = "Please select at least one symptom."

    if error:
        return jsonify({"error": error}), 400
    return jsonify({"diagnoses": diagnosis_results})

# --- Colon Cancer Model Loading and Prediction ---

colon_model = None
try:
    colon_model = load_model('colon_cancer (1).h5')
    print("Colon cancer model loaded successfully!")
except Exception as e:
    print(f"Error loading colon cancer model: {str(e)}")

colon_class_names = ['Colon_Adenocarcinoma', 'Colon_Benign_Tissue', 'Lung_Adenocarcinoma', 'Lung_Benign_Tissue', 'Lung_Squamous_Cell_Carcinoma']

def preprocess_colon_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = datagen.flow(img_array, batch_size=1).__next__()
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing colon image: {str(e)}")

@app.route('/colon', methods=['POST'])
def predict_colon():
    if colon_model is None:
        return jsonify({'error': 'Colon cancer model not loaded properly'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        
        try:
            Image.open(io.BytesIO(img_bytes))
        except:
            return jsonify({'error': 'Invalid image file'}), 400

        input_data = preprocess_colon_image(img_bytes)
        
        
        prediction = colon_model.predict(input_data)[0]
        predicted_class = int(np.argmax(prediction))

        
        return jsonify({
            'predicted_class_name': colon_class_names[predicted_class],
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 