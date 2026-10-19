from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200

VALID_STUDENT = {
    "school": "GP", "sex": "F", "age": 18, "address": "U", "famsize": "GT3",
    "Pstatus": "A", "Medu": 4, "Fedu": 4, "Mjob": "at_home", "Fjob": "teacher",
    "reason": "course", "guardian": "mother", "traveltime": 2, "studytime": 2,
    "failures": 0, "schoolsup": "yes", "famsup": "no", "paid": "no",
    "activities": "no", "nursery": "yes", "higher": "yes", "internet": "no",
    "romantic": "no", "famrel": 4, "freetime": 3, "goout": 4, "Dalc": 1,
    "Walc": 1, "health": 3, "absences": 6, "G1": 5,
}

def test_predict_valid_student():
    response = client.post("/predict", json=VALID_STUDENT)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {0,1}
    assert 0 <= data["probability"] <= 1

def test_predict_rejects_invalid_age():
    bad_student = {**VALID_STUDENT, "age": 30}
    response = client.post("/predict", json=bad_student)
    assert response.status_code == 422