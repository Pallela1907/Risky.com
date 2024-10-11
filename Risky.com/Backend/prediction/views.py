from django.shortcuts import render
from django.http import JsonResponse
import pickle
import json
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import warnings
import joblib

warnings.filterwarnings("ignore")

def encoding(value):
    if value == 'TIER1':
        return 0
    elif value == 'TIER2':
        return 1
    elif value == 'TIER3':
        return 2
    elif value == 'TIER4':
        return 3
    elif value == 'MC':
        return 0
    elif value == 'MO':
        return 1
    elif value == 'RETOP':
        return 2
    elif value == 'SC':
        return 3
    elif value == 'TL':
        return 4
    elif value == 'FEMALE':
        return 0
    elif value == 'MALE':
        return 1
    elif value == 'HOUSEWIFE':
        return 0
    elif value == 'PENS':
        return 1
    elif value == 'SAL':
        return 2
    elif value == 'SELF':
        return 3
    elif value == 'STUDENT':
        return 4
    elif value == 'Z':
        return 5
    elif value == 'NO':
        return 0
    elif value == 'OWNED BY OFFICE':
        return 1
    elif value == 'OWNED':
        return 2
    elif value == 'RENT':
        return 3

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            model = None
            print(data['features'])
            #print(data['features'])
            with open('C:/Users/sarat/Desktop/Tvs credit 6.0/Frontend/Backend/prediction/model.pkl', 'rb') as file:
                #model = pickle.load(file)
                model = joblib.load(file)
            #print(model)
            features = np.array(data['features']).reshape(1, -1)
            print(features)
            prediction = model.predict(features)
            prob = model.predict_proba(features)
            return JsonResponse(
                {
                    'prediction': prediction.tolist(),
                    'probability': prob.tolist(),
                    'details': {
                        'name': data['name'],
                        'tenure': data['tenure'],
                        'bouncedFirstEMI': data['bouncedFirstEMI'],
                        'maxMOB': data['maxMOB'],
                        'dealersCode': data['dealerCodes'],
                        'employmentType': data['employmentType'],
                        'gender': data['gender'],
                        'loanAmount': data['loanAmount'],
                        'employmentType': data['employmentType'],
                        'bounced12months': list(data['features'])[0],
                        'bouncedRepaying': list(data['features'])[1],
                        'ageLoanTaken': list(data['features'])[2],
                        'pastDue30': list(data['features'])[3],
                        'pastDue60': list(data['features'])[4],
                        'pastDue90': list(data['features'])[5],
                    }
                }
            )
        except Exception as e:
            print(e)
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def welcome(request):
    return JsonResponse({'message': 'Welcome to TVS!'})