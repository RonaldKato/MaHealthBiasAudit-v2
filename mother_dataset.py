"""
MOTHER Dataset - Maternal Health Survey Data
Contains survey responses in English, Luganda, and Runyankole-Rukiga
"""

from typing import Dict, List, Optional


class MotherDataset:
    """MOTHER dataset loader"""
    
    def __init__(self):
        self.data = None
    
    def load_your_data(self) -> Dict:
        """Load MOTHER dataset exactly as provided"""
        
        # ============================================================
        # Category 1: English Questions and Answers
        # ============================================================
        about_english_questions = [
            "Please state your location (Village) …………………………….",
            "What is your age?",
            "What is your highest level of education?",
            "Which language do you primarily speak at home?",
            "What is the age of your pregnancy?",
            "How many times have you been pregnant?",
            "How many children have you given birth to?",
            "Did you have any complications during your pregnancy?",
            "How many times have you experienced a miscarriage?",
            "How many times have you experienced a stillbirth?",
            "Did you receive vaccination during your recent pregnancy?",
            "How far do you travel to access antenatal care services?",
            "How many antenatal care visits have you attended?",
            "What is your primary mode of transportation to antenatal care?",
            "Where do you currently get most of your pregnancy-related information?",
            "What type of format do you prefer to get your information?",
            "In which language do you prefer to receive healthcare information?",
            "How often do you feel that the information you receive is understandable?",
            "What are the biggest barriers to attending antenatal visits for you?",
            "How comfortable do you feel discussing pregnancy concerns with male providers?",
            "Do you have access to a Mobile/Cell phone?",
            "Have you ever used a mobile app for pregnancy information?",
            "How likely would you be to use a mobile app in your local language?",
            "How important is information in your native language?",
            "How satisfied are you with current information sources?"
        ]
        
        answers_english = {
            'M-AN-015': [
                "Rwentuha", "26–35 years", "Secondary education", "Runyankore-Rukiga",
                "9 months", "2-3", "2 – 3", "No", "None", "None",
                "No", "Less than 5 km", "2-4 visits", "Walking",
                "Family and friends,Community health workers (VHTs),Radio", "Video",
                "Runyankore-Rukiga", "Sometimes", "Other: None", "Very comfortable",
                "No", "", "Unlikely", "Extremely important", "Satisfied"
            ],
            'K-ME-M0-006': [
                "Katete", "18–25 years", "Secondary education", "Runyankore-Rukiga",
                None, "4 – 5", "1", "No", "1", "2 – 4",
                "Yes", "5–10 km", "2-4 visits", "Walking",
                "Family and friends", "Audio", "Runyankore-Rukiga",
                "Most of the time", "Language barrier", None,
                "No", "No", "Very likely", "Very important", "Satisfied"
            ],
            'K-ME-MO-18': [
                "Kikuuto", "18–25 years", "Primary education", "Runyankore-Rukiga",
                "7 months", "1", "1", "No", "None", "None",
                "Yes", "5–10 km", "2-4 visits", "Public transport",
                "Family and friends,Radio", "Audio", "Runyankore-Rukiga",
                "Most of the time", "Distance/transportation", "Very comfortable",
                "No", None, "Neutral", "Extremely important", "Very satisfied"
            ]
        }
        
        # ============================================================
        # Category 2: Luganda Questions and Answers
        # ============================================================
        about_luganda_questions = [
            "Tukusaba okuwandika ekyalo (Village) kyobeeramu …………………………….",
            "Wa wofuna amawurire agakwata ku mbuto?",
            "Olina emyaka Mekka?",
            "Lulimi ki lwosinga okwogera nga oli wakka?",
            "Wakabeera lubuto emirundi emmeka?",
            "Obuyigirize bwo obwa waggulu kye ki?",
            "Ozadde/wazala abaana bameka?",
            "Wagenda mu ddwaliro emirundi emekka kukebeza olubuto lwo olwasembayo?",
            "Oli mumativu n'ebintu ebiriwo mu kiseera kino ebikwata ku nsonga z'olubuto by'olina?",
            "Offuna esimu?",
            "Kikulu kwenkana wa okumanya ebikwata ku lubuto okuweebwa mu lulimi lwo?",
            "Oyinza otya okukozesa pulogulaamu y'oku ssimu ewa ebikwata ku lubuto mu lulimi lwo?"
        ]
        
        answers_luganda = {
            'MBR/NP/024': [
                "Lyantonde", "Abasawo mu mabaddwaliro,Redio,TV", "Emyaka 18–25",
                "Orunyankore-Rukiga", "2-3", "Essomo lya pulayimale", "2-3",
                "2-4", "Ndi mumativu", "Iye", "Kikulu nnyo ddala", "Nsobola"
            ],
            'MBR/NP/023': [
                "Kagando - Kasese", "Abafamire n'emikwano,Abawereza b'okukyalo",
                "Emyaka 26–35", "Endala (wandiika wanoo)", "2-3",
                "Essomo lya wagulu (ettendekelo/yunivasite)", "2-3", "2-4",
                "Ndi mumativu", "Iye", "Kikulu nnyo ddala", "Nsobolera ddala"
            ],
            'MBR/NP/017': [
                "Nyamityobora", "Abawereza b'okukyalo,Abasawo mu mabaddwaliro,Redio",
                "Emyaka 36–45", "Oluganda", "4-5", "Siniya", "2-3", "2-4",
                "Ndi mumativu nnyo", "Iye", "Kikulu nnyo ddala", "Nsobolera ddala"
            ]
        }
        
        # ============================================================
        # Category 3: Runyankole-Rukiga Questions and Answers
        # ============================================================
        about_runyankole_rukiga_questions = [
            "Nyaburawe taho ekyaro kyaawe ……………………",
            "Nogyenda orugyendo rurikwinganaki waba nooza kutunga obuheereza bw'okukyebeza enda?",
            "Obwegyese bwawe obw'ahaiguru nibuha?",
            "Oine emyaka engahi?",
            "Ni rurimi ki oru orikukira kugamba omuka?",
            "Wahisa emirundi engahe orikutwara enda?",
            "Nokunda kutunga amakuru g'eby'amagara omu rurimi ki?",
            "Ozaire abaana bangahi?",
            "N'emirundi engahi ei orugirwe enda y'omwana ofiire?",
            "Okagyemesibwa omu bwire bu wabiire oine enda?",
            "Nokira kukozesa ntambura ki waba nooza ahairwariri kuyebeza enda?",
            "Ni mirundi engahi ei otaayaayiire eirwariro kukyebeza enda?",
            "Noobaasa kukoresa ota puroguraamu y'aha simu erikukuha amakuru agarikukwata aha kuzaara omu rurimi rwawe?",
            "Nohurira oshemereirwe/otiine kwerarikirira kugaaniira n'abashaho b'abashaija?"
        ]
        
        answers_runyankole_rukiga = {
            'MBR-AD-001': [
                "Katete", "Ahansi ya kilomita 5", "Obwegyese bwa yunivasite/eitendekyero",
                "Emyaka 18–25", "Orunyankore-Rukiga", "2-3", "Orunyankore-Rukiga",
                "2-3", "Ngaaha", "Eego", "Entambura y'abantu boona (taksi)",
                "Emirundi 5-7", "Nikibaasika munonga", "Nimba nyikaikiine munonga"
            ],
            'MBR-AD-002': [
                "Kamukuzi", "Kilomita 5–10", "Obwegyese bwa siniya",
                "Emyaka 18–25", "Orunyankore-Rukiga", "2-3", "Orunyankore-Rukiga",
                "2-3", "Ngaaha", "Ngaaha", "Entambura y'abantu boona (taksi)",
                "Emirundi 2-4", "Nikibaasika munonga", "Nimba ntaine kwikikana kwingi"
            ],
            'MBR-AD-003': [
                "KIBINGO", "Kilomita 5–10", "Obwegyese bwa siniya",
                "Emyaka 18–25", "Orunyankore-Rukiga", "2-3", "Orunyankore-Rukiga",
                "2-3", "Ngaaha", "Eego", "Entambura y'abantu boona (taksi)",
                "Emirundi 2-4", "Nikibaasika munonga", "Nimba nyikaikiine munonga"
            ]
        }
        
        # ============================================================
        # Returning the data as a dictionary
        # ============================================================
        self.data = {
            "about_english": {
                "questions": about_english_questions,
                "answers": answers_english
            },
            "about_luganda": {
                "questions": about_luganda_questions,
                "answers": answers_luganda
            },
            "about_runyankole_rukiga": {
                "questions": about_runyankole_rukiga_questions,
                "answers": answers_runyankole_rukiga
            }
        }
        return self.data