def load_your_data(self) -> Dict:
    """Load YOUR dataset exactly as provided"""
    
    # ============================================================
    # Category 1: English
    # ============================================================
    about_english = [
        "Please state your location (Village) ……………………………. ",
        "What is your age?",
        "What is your highest level of education?",
        "Which language do you primarily speak at home?",
        "What is the age of your pregnancy?",
        "How many times have you been pregnant?",
        "How many children have you given birth to?",
        "Did you have any complications during your pregnancy?",
        "If yes (In 3 above), select the type(s) of complication",
        "How many times have you experienced a miscarriage?",
        "How many times have you experienced a stillbirth?",
        "Did you receive vaccination during your recent pregnancy?",
        "If Yes (In 6 above), which vaccines did you receive?",
        "Were the vaccines given at …",
        "How far do you travel to access antenatal care services?",
        "How many antenatal care visits have you attended during your current pregnancy?",
        "What is your primary mode of transportation to antenatal care?",
        "Where do you currently get most of your pregnancy-related information?",
        "What type of format do you prefer to get your information?",
        "In which language do you prefer to receive healthcare information?",
        "How often do you feel that the information you receive from healthcare providers is understandable?",
        "What are the biggest barriers to attending antenatal visits for you?",
        "How comfortable do you feel discussing pregnancy-related concerns with male healthcare providers?",
        "Do you have access to a Mobile/Cell phone?",
        "Have you ever used a mobile app or online platform to get pregnancy-related information?",
        "How likely would you be to use a mobile app that provides pregnancy-related information in your local language?",
        "How important is it for pregnancy-related information to be provided in your native language?",
        "How satisfied are you with the current sources of pregnancy-related information available to you?"
    ]
    
    answers_english = {
        'M-AN-015': [
            "Rwentuha", "26–35 years", "Secondary education", "Runyankore-Rukiga",
            "9 months", "02-Mar", "2 – 3", "No", "Infection", "None", "None",
            "No", "", "9 months", "Less than 5 km", "2-4 visits", "Walking",
            "Family and friends,Community health workers (VHTs),Radio", "Video",
            "Runyankore-Rukiga", "Sometimes", "Other: None", "Very comfortable",
            "No", "", "Unlikely", "Extremely important", "Satisfied"
        ],
        'K-ME-M0-006': [
            "Katete", "18–25 years", "Secondary education", "Runyankore-Rukiga",
            None, "4 – 5", "1", "No", "High blood pressure", "1", "2 – 4",
            "Yes", "Pertussis (Whooping cough)", "6 weeks", "5–10 km",
            "2-4 visits", "Walking", "Family and friends", "Audio",
            "Runyankore-Rukiga", "Most of the time", "Language barrier", None,
            "No", "No", "Very likely", "Very important", "Satisfied"
        ],
        'K-ME-MO-18': [
            "Kikuuto", "18–25 years", "Primary education", "Runyankore-Rukiga",
            "7 months", "1", "1", "No", None, "None", "None", "Yes", "Tetanus",
            None, "5–10 km", "2-4 visits", "Public transport",
            "Family and friends,Radio", "Audio", "Runyankore-Rukiga",
            "Most of the time", "Distance/transportation", "Very comfortable",
            "No", None, "Neutral", "Extremely important", "Very satisfied"
        ],
        'K-ME-MO-020': [
            "Kabwohe T.C", "18–25 years", "Secondary education",
            "Runyankore-Rukiga,Luganda", "3 months", "1", None, "No", None,
            "None", "None", "No", None, None, "Less than 5 km", "2-4 visits",
            "Public transport", "Family and friends,Healthcare providers at clinics/hospitals",
            "Audio", "Luganda,English", "Most of the time",
            "Lack of the right information", "Somewhat comfortable", "Yes",
            "No", "Neutral", "Extremely important", "Neutral"
        ],
        'CM-MBR-01-011': ["26–35 years"],
        'K-ME-M0-19': [
            "Kabwohe T.C", "36–45 years", "Tertiary education", "Runyankore-Rukiga",
            "5 months", "More than 5", "4 – 5", "Yes", "Bleeding", "None",
            "None", "Yes", "Tetanus,Hepatitis B", "6 weeks,14 weeks", "5–10 km",
            "2-4 visits", "Public transport", "Healthcare providers at clinics/hospitals",
            "Audio", "Runyankore-Rukiga", "Sometimes",
            "Distance/transportation,Cost", "Somewhat uncomfortable", "Yes", "No",
            "Likely", "Extremely important", "Satisfied"
        ],
        'K-JK-MO-001': [
            "Nyakashambya", "26–35 years", "Tertiary education",
            "Runyankore-Rukiga,Other: ", "already delivered", "02-Mar", "2 – 3",
            "Yes", "Bleeding", "None", "None", "Yes", "Polio,Tetanus",
            "6 weeks,At birth", "Less than 5 km", "2-4 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga,English", "Most of the time",
            "Distance/transportation,Other: ", "Somewhat comfortable", "Yes", "No",
            "Likely", "Extremely important", "Satisfied"
        ],
        'K-JK-MO-002': [
            "Kabwohe", "26–35 years", "Tertiary education", "Other: ",
            "post partum", "1", "1", "No", None, "None", "None", "No", None,
            None, "Less than 5 km", None, "Walking", None, "Video", "English",
            "Always", "Language barrier", "Very comfortable", "Yes", "Yes",
            "Likely", "Very important", None
        ],
        'K-JK-MO-003': [
            "Rushoroza", "36–45 years", "Secondary education", "Runyankore-Rukiga",
            "post mortam", "4 – 5", "4 – 5", "Yes", "Other (Please specify)",
            "None", "None", "Yes", "Tetanus", "At birth,6 weeks,10 weeks",
            "Less than 5 km", "5-7 visits", "Public transport",
            "Healthcare providers at clinics/hospitals,Radio,Television,Online platforms (Facebook groups",
            "Audio,Video", "Luganda", "Always", "Cost", "Very comfortable",
            "Yes", "Yes", "Very likely", "Very important", "Satisfied"
        ],
        'K-JK-MO-004': [
            "Rutooma", "18–25 years", "Primary education", "Luganda", "8 months",
            "02-Mar", "1", "No", None, "1", "None", "Yes", "Tetanus", None,
            "Less than 5 km", "8 visits or more", "Walking",
            "Healthcare providers at clinics/hospitals", "Text", "Luganda",
            "Most of the time", None, "Somewhat uncomfortable", "Yes", "No",
            "Unlikely", "Very important", "Satisfied"
        ],
        'K-JK-MO-005': [
            "Rutoma", "26–35 years", "Tertiary education", "Other: ",
            "4.2 months", "4 – 5", "2 – 3", "Yes", "Infection", "None", "1",
            "Yes", "Tetanus", "At birth", "Less than 5 km", "2-4 visits",
            "Walking", "Healthcare providers at clinics/hospitals", "Video",
            "Other: ", "Most of the time", None, "Very comfortable", "Yes", "No",
            None, "Extremely important", "Satisfied"
        ],
        'K-JK-MO-006': [
            "Rutooma", "26–35 years", "Tertiary education", "Other: lukonjo",
            "one month", "02-Mar", "2 – 3", "No", None, "None", "None", "Yes",
            "Tetanus", "At birth", "Less than 5 km", "1 visit", "Walking",
            "Community health workers (VHTs)", "Video", "Other: lukonjo",
            "Most of the time", "Distance/transportation", "Very comfortable",
            "Yes", "Yes", "Unlikely", "Very important", "Satisfied"
        ],
        'K-JK-MO-007': [
            "Ishekye", "36–45 years", "Primary education", "Luganda", "3 months",
            "More than 5", "2 – 3", "Yes", "Miscarriage/Stillbirth", "2 – 4",
            "None", "Yes", "Tetanus",
            "At birth,6 weeks,10 weeks,14 weeks,6 months,9 months,1 year",
            "Less than 5 km", "2-4 visits", "Walking",
            "Community health workers (VHTs),Healthcare providers at clinics/hospitals",
            "Audio", "Luganda", "Most of the time", "Yes", "Satisfied",
            "Most of the time", "No", "Very likely",
            "Cost,Distance/transportation", "Extremely important"
        ],
        'K-JK-MO-008': [
            "Kabwohe", "18–25 years", "Secondary education", "Runyankore-Rukiga",
            "N/A", "1", "1", "Yes", "High blood pressure", "None", "None",
            "Yes", "Tetanus", None, "Less than 5 km", "2-4 visits",
            "Walking,Public transport",
            "Family and friends,Community health workers (VHTs)", "Audio",
            "Runyankore-Rukiga,Other: English", "Always", None, "Very comfortable",
            "No", "No", "Unlikely", "Extremely important", "Very satisfied"
        ],
        'K-JK-MO-009': [
            "Kabwohe", "26–35 years", "Secondary education", "Runyankore-Rukiga",
            "2 months", "02-Mar", "2 – 3", "Yes", "Infection", "None", "None",
            "Yes", "Tetanus", "At birth,6 weeks,10 weeks,14 weeks,6 months,9 months,1 year",
            "5–10 km", "2-4 visits", "Public transport",
            "Healthcare providers at clinics/hospitals,Radio,Television", "Text",
            "Runyankore-Rukiga", "Most of the time",
            "Cost,Distance/transportation", "Very uncomfortable", "Yes", "Yes",
            "Very likely", "Extremely important", "Dissatisfied"
        ],
        'K-JK-MO-010': [
            "Kyamatongo-Nyanga ward", "26–35 years", "Tertiary education",
            "Runyankore-Rukiga", "4 months", "02-Mar", "2 – 3", "No", None,
            "None", "None", "Yes", "Tetanus", "At birth", "5–10 km",
            "2-4 visits", "Public transport", "Community health workers (VHTs)",
            "Text", "Runyankore-Rukiga", "Most of the time", "Distance/transportation",
            "Very comfortable", "Yes", "No", "Neutral", "Extremely important",
            "Satisfied"
        ],
        'K-JK-MO-014': [
            None, "18–25 years", "Tertiary education", "Runyankore-Rukiga",
            "32 weeks", "1", "1", "Yes", "Other (Please specify)", "None",
            "None", "No", None, None, "Less than 5 km", "5-7 visits", "Walking",
            "Family and friends,Healthcare providers at clinics/hospitals,Other: tiktok, whatsapp",
            "Video", "Runyankore-Rukiga,English", "Always",
            "Lack of the right information", "Very comfortable", "Yes", "Yes",
            "Unlikely", "Extremely important", "Neutral"
        ],
        'K-JK-MO-013': [
            "Sheema town", "18–25 years", "Tertiary education", "Runyankore-Rukiga",
            "2 months", "1", "1", "Yes", "Other (Please specify)", "None",
            "None", "No", None, None, "Less than 5 km", "1 visit",
            "Public transport", "Family and friends,Online platforms (Facebook groups",
            "Audio", "English", "Most of the time", "Cost", "Very comfortable",
            "Yes", "No", "Neutral", "Important", "Dissatisfied"
        ],
        'K-JK-MO-018': [
            "Kigimbi", "26–35 years", "Tertiary education", "Runyankore-Rukiga",
            "2 months post portum", "4 – 5", "2 – 3", "Yes", "Miscarriage/Stillbirth",
            "2 – 4", "None", "Yes", "Tetanus", "At birth,6 weeks",
            "Less than 5 km", "8 visits or more", "Public transport",
            "Community health workers (VHTs),Healthcare providers at clinics/hospitals,Radio",
            "Text", "Runyankore-Rukiga,English", "Always", "Cost",
            "Very comfortable", "Yes", None, "Very likely", "Very important",
            "Satisfied"
        ],
        'K-JK-MO-015': [
            "Kabwohe hill", "18–25 years", "Tertiary education", "Runyankore-Rukiga",
            "7 months", "1", "1", "Yes", "Infection", "None", None, "Yes",
            "Tetanus", "6 months", "Less than 5 km", "2-4 visits", "Walking",
            "Family and friends", "Text", "English", "Always",
            "Availability of healthcare providers", "Very comfortable", "Yes",
            "Yes", "Unlikely", "Extremely important", "Very satisfied"
        ],
        'K-JK-MO-016': [
            "Kigarama", "18–25 years", "Primary education", "Runyankore-Rukiga",
            "five month", "1", None, "No", None, "None", "None", "No", None,
            None, "Less than 5 km", "1 visit", "Public transport",
            "Family and friends", "Audio", "Runyankore-Rukiga", None, None,
            "Neutral", None, "No", "Very likely", "Extremely important",
            "Satisfied"
        ],
        'K-JK-MO-017': [
            "Rushoroza, Nyanga", "26–35 years", "Tertiary education",
            "Runyankore-Rukiga", "6 months", "1", "1", "No", None, "None",
            "None", "Yes", "Tetanus", "14 weeks", "Less than 5 km", "2-4 visits",
            "Public transport", "Healthcare providers at clinics/hospitals",
            "Text", "Runyankore-Rukiga", "Most of the time",
            "Availability of healthcare providers", "Very comfortable", "Yes",
            "Yes", "Likely", "Extremely important", "Satisfied"
        ],
        'MBR/NP/025': [
            "Biharwe", "36–45 years", "No formal education", "Runyankore-Rukiga",
            "5 months", "1", None, "Yes", "Infection", "2 – 4", "None", "Yes",
            "Tetanus", "10 weeks", "5–10 km", "2-4 visits", "Public transport",
            "Community health workers (VHTs),Healthcare providers at clinics/hospitals,Radio",
            "Audio", "Runyankore-Rukiga", "Sometimes",
            "Cost,Distance/transportation,Language barrier,Availability of healthcare providers,Lack of the right information",
            "Somewhat comfortable", "Yes", "No", "Very likely", "Extremely important",
            "Satisfied"
        ],
        'MBR/NP/013': [
            "Kitebero-Mbarara", "26–35 years", "Secondary education",
            "Runyankore-Rukiga", "delivered", "02-Mar", "2 – 3", "No", None,
            "None", "None", "Yes", "Tetanus,Hepatitis B", "At birth",
            "Less than 5 km", "2-4 visits", "Walking",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga", "Most of the time", "Cost", "Very comfortable",
            "Yes", None, "Very likely", "Very important", "Satisfied"
        ],
        'MBR/NP/012': [
            "Kakigani", "26–35 years", "Secondary education", "Runyankore-Rukiga",
            "9 months", "02-Mar", "1", "No", None, "None", "None", "Yes",
            "Tetanus", None, "Less than 5 km", "2-4 visits", "Walking",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga", "Always", "Cost", "Very comfortable", "No",
            "No", "Neutral", "Extremely important", "Satisfied"
        ],
        'MBR/NP/009': [
            "Kabwohe", "26–35 years", "Tertiary education", "Runyankore-Rukiga",
            "8 months", "02-Mar", "2 – 3", "No", None, "None", "None", "Yes",
            "Tetanus", None, "Less than 5 km", "2-4 visits", "Other: boda boda",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga,English", "Most of the time",
            "Availability of healthcare providers", "Somewhat comfortable", "Yes",
            "No", "Very likely", "Extremely important", "Satisfied"
        ],
        'MBR/NP/008': [
            "Bweyegamba - sheema", "26–35 years", "Secondary education",
            "Runyankore-Rukiga", "9", "4 – 5", "2 – 3", "No", None, "None",
            "None", "Yes", "Tetanus", None, "Less than 5 km", "5-7 visits",
            "Walking", "Healthcare providers at clinics/hospitals", "Video",
            "Runyankore-Rukiga", "Most of the time", "Other: not feeling pain",
            "Somewhat comfortable", "No", "No", "Neutral", "Extremely important",
            "Satisfied"
        ],
        'MBR/NP/007': [
            "Lugazi-Mbarara", "18–25 years", "Secondary education",
            "Runyankore-Rukiga", "9 months", "02-Mar", "2 – 3", "No", None,
            "None", "None", "Yes", "Tetanus", None, "Less than 5 km",
            "2-4 visits", "Public transport",
            "Family and friends,Community health workers (VHTs),Healthcare providers at clinics/hospitals,Radio",
            "Video", "Runyankore-Rukiga", "Always",
            "Other: feeling weak, rude workers", "Very comfortable", "Yes", "Yes",
            "Very likely", "Extremely important", "Very satisfied"
        ],
        'MBR/NP/005': [
            "Mbarara town", "26–35 years", "Secondary education",
            "Runyankore-Rukiga", "8 months", "1", None, "Yes",
            "Other (Please specify)", "None", "None", "Yes", "Tetanus", None,
            "Less than 5 km", "5-7 visits", None,
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga,English", "Most of the time",
            "Availability of healthcare providers", "Somewhat comfortable", "Yes",
            None, "Very likely", "Extremely important", "Satisfied"
        ],
        'M/AN/09': [
            "Nyamitanga", "26–35 years", "Tertiary education", "Runyankore-Rukiga",
            "6 months", "02-Mar", "1", "No", None, "None", "None", "Yes",
            "Tetanus", "6 months", "Less than 5 km", "2-4 visits",
            "Other: boda,Public transport",
            "Family and friends,Healthcare providers at clinics/hospitals",
            "Video", "Runyankore-Rukiga,English", "Always",
            "Distance/transportation", "Very comfortable", "Yes", "No",
            "Very likely", "Very important", "Satisfied"
        ],
        'MBR/NP/004': [
            "Kasana-birere", "18–25 years", "Primary education",
            "Runyankore-Rukiga", "8 months", "02-Mar", "2 – 3", "No", None,
            "None", "None", "Yes", "Tetanus", None, "Less than 5 km",
            "2-4 visits", "Public transport",
            "Community health workers (VHTs),Healthcare providers at clinics/hospitals,Radio",
            "Audio", "Runyankore-Rukiga", "Sometimes",
            "Availability of healthcare providers", "Very comfortable", "No",
            "No", "Neutral", "Extremely important", "Satisfied"
        ],
        'M/NP/011': [
            "Kamukuzi", "36–45 years", "Tertiary education", "Runyankore-Rukiga",
            "4 weeks", "02-Mar", "2 – 3", "Yes", "Infection", "None", "None",
            "Yes", "Tetanus", "6 weeks", "11–20 km", None, "Public transport",
            "Healthcare providers at clinics/hospitals,Radio,Television",
            "Video", "Runyankore-Rukiga", "Most of the time", "Distance/transportation",
            "Very comfortable", "Yes", "No", "Very likely", "Extremely important",
            "Satisfied"
        ],
        'M/AN/06': [
            "Nyakaizi", "26–35 years", "Secondary education", "Runyankore-Rukiga",
            "38 weeks", "02-Mar", "2 – 3", "No", None, "None", "None", "Yes",
            "Tetanus", "6 months", "5–10 km", "2-4 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Text", "English",
            "Most of the time", "Distance/transportation", None, "Yes", "Yes",
            "Likely", "Very important", "Very satisfied"
        ],
        'M/AN/05': [
            "Rwobuyenje", "26–35 years", "Tertiary education", "Luganda",
            "5 months", "4 – 5", "4 – 5", "No", None, "None", "None", "No",
            None, "14 weeks", "5–10 km", "1 visit", "Private vehicle",
            "Healthcare providers at clinics/hospitals", "Video", "English",
            "Most of the time", "Cost", "Very comfortable", "Yes", "Yes",
            "Very likely", "Extremely important", "Neutral"
        ],
        'M/AN/01': [
            "Ruharo-Mbarara", "26–35 years", "Tertiary education",
            "Runyankore-Rukiga", "8 months", "02-Mar", None, "Yes",
            "Other (Please specify)", "2 – 4", "2 – 4", "Yes", "Tetanus",
            "14 weeks", "Less than 5 km", "5-7 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga,English", "Most of the time",
            "Cost,Distance/transportation,Availability of healthcare providers",
            "Neutral", "No", "Yes", "Likely", "Very important", "Very satisfied"
        ],
        'M/AN/08': [
            "Nyakaizi- kakoba", "26–35 years", "Secondary education",
            "Runyankore-Rukiga", "9 months", "1", "1", "No", None, "None",
            "None", "Yes", "Tetanus", "6 months", "5–10 km", "5-7 visits",
            "Other: boda", "Family and friends,Healthcare providers at clinics/hospitals",
            "Video", "Runyankore-Rukiga,English", "Most of the time",
            "Distance/transportation", "Somewhat comfortable", "Yes", "No",
            "Likely", "Very important", "Satisfied"
        ],
        'M/AN/011': [
            "Kakoba", "18–25 years", "Tertiary education", "Runyankore-Rukiga",
            "16 weeks", "02-Mar", "1", "Yes", "Miscarriage/Stillbirth", "1",
            "None", "Yes", "Tetanus",
            "At birth,6 weeks,10 weeks,14 weeks,6 months,9 months,1 year",
            "Less than 5 km", "2-4 visits", "Public transport",
            "Healthcare providers at clinics/hospitals,Television,Online platforms (Facebook groups",
            "Video", "English", "Most of the time", None, "Neutral", "Yes",
            "Yes", "Very likely", "Somewhat important", "Neutral"
        ],
        'M/AN/10': [
            "Kamukuzi", "36–45 years", "Tertiary education", "Runyankore-Rukiga",
            "7 months", "1", "1", "Yes", None, "None", "None", "No", None,
            None, "Less than 5 km", "2-4 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Video",
            "Runyankore-Rukiga,English", "Most of the time", None, "Very comfortable",
            "Yes", "Yes", "Very likely", "Extremely important", "Very satisfied"
        ],
        'M/AN/22': [
            "Rubirizi", "26–35 years", "Primary education", "Runyankore-Rukiga",
            "9 months", "02-Mar", "2 – 3", "Yes", "Infection", "None", "None",
            "Yes", "Tetanus", "6 weeks,14 weeks", "5–10 km", "2-4 visits",
            "Public transport", "Healthcare providers at clinics/hospitals",
            "Audio,Video", "Runyankore-Rukiga", "Always", "Other: none",
            "Very comfortable", "Yes", None, "Likely", "Extremely important",
            "Very satisfied"
        ],
        'M/AN/24': [
            "Booma", "Above 45 years", "Tertiary education", "Runyankore-Rukiga",
            "7 months", "02-Mar", "2 – 3", "No", None, "None", "None", "Yes",
            "Tetanus", None, "Less than 5 km", "2-4 visits", "Other: boda boda",
            "Family and friends,Healthcare providers at clinics/hospitals,Television",
            "Audio", "Runyankore-Rukiga", "Most of the time",
            "Availability of healthcare providers,Other: spending a lot of time at the clinic",
            "Neutral", "Yes", "No", "Very likely", "Extremely important", None
        ],
        'KB008': [
            "Kazo-Kigorogoro", "26–35 years", "Secondary education",
            "Runyankore-Rukiga", "8 months", "02-Mar", "2 – 3", "Yes",
            "Infection", "None", "None", "Yes", "Tetanus", "14 weeks",
            "5–10 km", "5-7 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga", "Always", "Cost", "Very comfortable", "Yes",
            None, "Likely", "Important", "Satisfied"
        ],
        'KB007': [
            "Nyamityobora", "36–45 years", "Secondary education",
            "Runyankore-Rukiga", "1 month", "02-Mar", "2 – 3", "No", None,
            "None", "None", "No", None, None, "Less than 5 km", "1 visit",
            "Public transport", "Healthcare providers at clinics/hospitals",
            "Text", "Runyankore-Rukiga", "Most of the time", "Cost",
            "Somewhat comfortable", "No", "No", "Very likely", "Extremely important",
            "Neutral"
        ],
        'KB006': [
            "Kamukuzi", "36–45 years", "Primary education", "Runyankore-Rukiga",
            "2 month", "02-Mar", "2 – 3", "Yes", "Infection", "None", "None",
            "No", None, None, "11–20 km", "1 visit", "Public transport",
            "Healthcare providers at clinics/hospitals", "Video",
            "Runyankore-Rukiga", "Always", "Cost,Distance/transportation",
            "Very comfortable", "No", "No", "Likely", "Important", "Satisfied"
        ],
        'MBR-AD-011': [
            "Kacence", "26–35 years", "Primary education", "Runyankore-Rukiga",
            "2 months", "1", "1", "Yes", "Preterm labor", "None", "None",
            "Yes", "Tetanus", None, "Less than 5 km", "1 visit", "Walking",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga", "Always", "Cost", "Very comfortable", "Yes",
            "No", "Unlikely", "Extremely important", "Very satisfied"
        ],
        'MBR-AD-015': [
            "Rwobuyenje", "36–45 years", "Secondary education",
            "Runyankore-Rukiga", "32 weeks", "02-Mar", "1", "No", None,
            "None", "None", "Yes", "Tetanus", None, "5–10 km", "2-4 visits",
            "Public transport", "Community health workers (VHTs)", "Audio",
            "Runyankore-Rukiga", "Always", "Distance/transportation",
            "Very comfortable", "Yes", None, "Likely", "Very important", None
        ],
        'MBR-AD-016': [
            "Katete", "18–25 years", "Primary education", "Runyankore-Rukiga",
            "4 months", "02-Mar", "1", "No", None, "None", "None", "Yes",
            "Tetanus", "14 weeks", "Less than 5 km", "1 visit", "Public transport",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga", "Always", "Distance/transportation",
            "Very comfortable", "No", None, None, "Extremely important",
            "Very satisfied"
        ],
        'MBR-AD-017': [
            "Prisons", "36–45 years", "Secondary education", "Other: english",
            "16 weeks", "More than 5", "4 – 5", "No", None, "None", "None",
            "Yes", "Tetanus", "10 weeks", "Less than 5 km", "2-4 visits",
            "Walking", "Healthcare providers at clinics/hospitals", "Audio",
            "English", "Always", None, "Very comfortable", "Yes", None, None,
            "Extremely important", "Very satisfied"
        ],
        'MBR-AD-019': [
            "Nkokonjeru", "18–25 years", "Primary education", "Runyankore-Rukiga",
            "2", "02-Mar", None, "Yes", "Bleeding,Other (Please specify)", "1",
            "None", "Yes", "Tetanus", "14 weeks", "5–10 km", "2-4 visits",
            "Public transport", "Healthcare providers at clinics/hospitals",
            "Audio", "Runyankore-Rukiga", "Always", "Cost", "Very comfortable",
            "No", "No", "Very unlikely", "Extremely important", "Very satisfied"
        ],
        'MBR-AD-021': [
            "Lubiri", "26–35 years", "Secondary education", "Runyankore-Rukiga",
            "7 months", "02-Mar", "1", "No", None, "None", "None", "Yes",
            "Tetanus", "10 weeks", "5–10 km", "5-7 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga", "Most of the time", "Cost", "Very comfortable",
            "Yes", "No", "Likely", "Extremely important", "Very satisfied"
        ],
        'MBR-AD-022': [
            "Kagando", "18–25 years", "Secondary education", "Runyankore-Rukiga",
            "30 weeks", "02-Mar", "1", "No", None, "2 – 4", "None", "Yes",
            "Tetanus", "At birth", "Less than 5 km", "5-7 visits", "Public transport",
            "Radio", "Video", "Runyankore-Rukiga", "Always", "Cost",
            "Very comfortable", "Yes", "No", "Very likely", "Extremely important",
            "Very satisfied"
        ],
        'MBR-AD-023': [
            "Nyamuyanja", "36–45 years", "Secondary education", "Runyankore-Rukiga",
            "9 months", "4 – 5", "2 – 3", "No", None, "None", "None", "Yes",
            "Tetanus", "14 weeks", "More than 20 km", "5-7 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Audio",
            "Runyankore-Rukiga", "Always", "Cost,Distance/transportation",
            "Very comfortable", "No", None, None, None, None
        ],
        'KB011': [
            "Kitobero", "Above 45 years", "No formal education", "Runyankore-Rukiga",
            "4 months", "More than 5", "4 – 5", "Yes", "High blood pressure",
            "None", "None", "Yes", "Tetanus", "14 weeks", "Less than 5 km",
            "2-4 visits", "Walking", "Healthcare providers at clinics/hospitals",
            "Audio", "Runyankore-Rukiga", "Most of the time", "Cost",
            "Somewhat comfortable", "No", None, "Unlikely", "Extremely important",
            "Neutral"
        ],
        'KB013': [
            "Ngatunda", "26–35 years", "Primary education", "Luganda",
            "3 months", "1", "1", "No", None, "None", "None", "Yes", "Tetanus",
            "14 weeks", "11–20 km", "2-4 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Audio", None,
            "Always", "Distance/transportation", "Somewhat comfortable", "Yes",
            "Yes", "Likely", "Important", "Satisfied"
        ],
        'KB018': [
            "Nyamityobora", "26–35 years", "Secondary education", "Runyankore-Rukiga",
            "7 months", "02-Mar", "2 – 3", "No", None, "None", "None", "Yes",
            "Tetanus", "14 weeks", "11–20 km", "5-7 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Video",
            "Runyankore-Rukiga", "Always",
            "Cost,Distance/transportation,Language barrier,Availability of healthcare providers",
            "Very comfortable", "Yes", "No", "Likely", "Very important", "Satisfied"
        ],
        'KB015': [
            "Katete", "18–25 years", "Secondary education", "Runyankore-Rukiga",
            "N/A", "1", "1", "No", None, "None", "None", "Yes", "Tetanus",
            "14 weeks", "5–10 km", "5-7 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Audio", "English",
            "Always", "Cost,Availability of healthcare providers", "Neutral",
            "Yes", "No", "Likely", "Important", "Neutral"
        ],
        'KB014': [
            "Rwebikoona", "26–35 years", "Tertiary education", "Runyankore-Rukiga",
            "6 months", "1", None, "No", None, "None", "None", "Yes", "Tetanus",
            "14 weeks", "Less than 5 km", "2-4 visits", "Public transport",
            "Healthcare providers at clinics/hospitals", "Audio", "English",
            "Always", "Cost", "Very comfortable", "Yes", "No", "Likely",
            "Important", "Satisfied"
        ],
        'KB010': [
            "Ruharo", "18–25 years", "Secondary education", "Runyankore-Rukiga",
            "5 months", "1", None, "No", None, "None", "None", "No", None,
            None, "5–10 km", "1 visit", "Public transport",
            "Healthcare providers at clinics/hospitals", "Video",
            "Runyankore-Rukiga", "Most of the time", "Cost,Distance/transportation",
            "Somewhat comfortable", "No", "No", "Neutral", "Very important",
            "Satisfied"
        ],
        'KB016': [
            "Kakyiika", "26–35 years", "Primary education", "Runyankore-Rukiga",
            "3 month", "02-Mar", "1", "No", None, "None", "None", "Yes",
            "Tetanus", "14 weeks", "Less than 5 km", "1 visit", "Walking",
            "Healthcare providers at clinics/hospitals", "Video",
            "Runyankore-Rukiga", "Always", "Cost", "Very comfortable", "No",
            "No", "Likely", "Very important", "Satisfied"
        ],
    }
    
    # ============================================================
    # Category 2: Luganda
    # ============================================================
    about_luganda = [
        "Tukusaba okuwandika ekyalo (Village) kyobeeramu …………………………….",
        "Wa wofuna amawurire agakwata ku mbuto?",
        "Biseera ki byewagemesibwa …",
        "Osinga okwagala kufuna amawulire agakwata ku mbuto mu ngeli ki?",
        "Olina emyaka Mekka?",
        "Bwekiba nti iye (wagulu), bakugema ddagal lya kuziyiza bulwadde ki?",
        "Otambula olugendo olwenkanaki okufuna obuwereza bw’okukebeza olubuto?",
        "Bwekiba nti iye (mu namb. 3 wagulu), Londa ebyakusumbuwa",
        "Wakavaamu olubuto emirundi emmeka?",
        "Ntambuki gyosinga okukozesa ng’ogenda muddwaliro kukebeza olubuto?",
        "Wakazala abaana nga bafu emirundi emmekka?",
        "Wagemesebwa mu biseera byewali lubuto naddala olwakasembayo?",
        "Lulimi ki lwosinga okwogera nga oli wakka?",
        "Wakabeera lubuto emirundi emmeka?",
        "Obuyigirize bwo obwa waggulu kye ki?",
        "Mirindi ki gyowulira nti amawulire gofunaokuva mubasawo gategerekeka?",
        "Ozadde/wazala abaana bameka?",
        "Wagenda mu ddwaliro emirundi emekka kukebeza olubuto lwo olwasembayo?",
        "Oli mumativu kwenkana wa n'ebintu ebiriwo mu kiseera kino ebikwata ku nsonga z'olubuto by'olina?",
        "Wali okozesezako app oba omutimbagano to kufuna amawulire agakwatagana n’olubuto?",
        "Offuna esimu?",
        "Kikulu kwenkana wa okumanya ebikwata ku lubuto okuweebwa mu lulimi lwo?",
        "Oyinza otya okukozesa pulogulaamu y'oku ssimu ewa ebikwata ku lubuto mu lulimi lwo?",
        "Engeri gy'owulira obulungi ng'oyogera n'abasawo abasajja ku nsonga z'olubuto?"
    ]
    
    answers_luganda = {
        'MBR/NP/024': [
            "Lyantonde", "Abasawo mu mabaddwaliro,Redio,TV", "Wenazalira,Wiki 6",
            "Mu maloboozi", "Emyaka 18–25", "Tetanus", "Kilomita ezisuba 20",
            "Puresa", "Sivaangamu lubuto", "Motoka zabantu bonna",
            "Sizaangako omwana omufu", "Iye", "Orunyankore-Rukiga", "02-Mar",
            "Essomo lya pulayimale", "Ebiseera ebisinga", "02-Mar", "02-Apr",
            "Ndi mumativu", "Nedda", "Iye", "Kikulu nnyo ddala", "Nsobola",
            "Mpulira bulungi nnyo"
        ],
        'MBR/NP/023': [
            "Kagando - Kasese",
            "Abafamire n’emikwano,Abawereza b’okukyalo (VHTs),Abasawo mu mabaddwaliro,Redio",
            "Wiki 14", "Mu maloboozi", "Emyaka 26–35", "Tetanus", "Kilomita 5–10",
            "Oburwadde obusaasana,Sukari", "1", "Motoka zabantu bonna",
            "Sizaangako omwana omufu", "Iye", "Endala (wandiika wanoo)", "02-Mar",
            "Essomo lya wagulu (ettendekelo/yunivasite)", "Ebiseera ebisinga",
            "02-Mar", "02-Apr", "Ndi mumativu", "Iye", "Iye", "Kikulu nnyo ddala",
            "Nsobolera ddala", "Mpulira bulungi nnyo"
        ],
        'MBR/NP/017': [
            "Nyamityobora",
            "Abawereza b’okukyalo (VHTs),Abasawo mu mabaddwaliro,Redio",
            None, "Mu maloboozi", "Emyaka 36–45", "Tetanus", "Kilomita 5–10",
            None, "02-Apr", "Motoka zabantu bonna", "Sizaangako omwana omufu",
            "Iye", "Oluganda", "04-May", "Siniya", "Gyonna", "02-Mar", "02-Apr",
            "Ndi mumativu nnyo", "Nedda", "Iye", "Kikulu nnyo ddala",
            "Nsobolera ddala", "Mpulira bulungi nnyo"
        ],
        'MBR/NP/015': [
            "Kiba - Bubare",
            "Abafamire n’emikwano,Abawereza b’okukyalo (VHTs),Abasawo mu mabaddwaliro,Redio",
            None, "Mu maloboozi", "Emyaka 26–35", "Tetanus", "Kilomita 11–20",
            "Oburwadde obusaasana", "1", "Motoka zabantu bonna", "1", "Iye",
            "Orunyankore-Rukiga", "02-Mar", "Essomo lya pulayimale",
            "Ebiseera ebisinga", "02-Mar", "02-Apr", "Ndi mumativu", "Nedda",
            "Nedda", "Kikulu nnyo ddala", "Nsobolera ddala", "Mpulira bulungi nnyo"
        ],
        'MBR/NP/014': [
            "Sanga", "Abasawo mu mabaddwaliro,Redio,TV", None, "Mu maloboozi",
            "Emyaka 26–35", "Tetanus", "Kilomita 11–20", None, "1",
            "Motoka zabantu bonna", "Sizaangako omwana omufu", "Iye",
            "Orunyankore-Rukiga", "04-May", "Siniya", "Gyonna", "02-Mar", "02-Apr",
            "Ndi mumativu", "Nedda", "Iye", "Kikulu nnyo ddala", "Nsobolera ddala",
            "Mpulira bulungi nnyo"
        ],
        'MBR/NP/016': [
            "Kitebero",
            "Abawereza b’okukyalo (VHTs),Abasawo mu mabaddwaliro,Redio",
            None, "Mu maloboozi", "Emyaka 26–35", "Tetanus", "Kilomita 5–10",
            "Oburwadde obusaasana", "Sivaangamu lubuto", "Okutambula",
            "Sizaangako omwana omufu", "Iye", "Orunyankore-Rukiga", "02-Mar",
            "Siniya", "Ebiseera ebisinga", "02-Mar", "05-Jul", "Ndi mumativu",
            "Nedda", "Iye", "Kikulu nnyo ddala", "Nsobola", "Mpulira bulungi nnyo"
        ],
        'K-ME-012': [
            "Kamwezi B", None, None, None, "Emyaka 18–25", None, "Kilomita 5–10",
            "Okuleeat omusaayi", None, None, None, "Iye", None, None,
            "Siniya", "Ebiseera ebisinga", "02-Mar", "05-Jul", "Ndi mumativu",
            "Nedda", "Nedda", "Kikulu nnyo ddala", "Simanyi", "Mpulira bulungi nnyo"
        ],
        'K-ME-MO-11': [
            "Kabwohe hill - kabwohe ward", None, None, None, "Emyaka 36–45",
            None, "Kilomita 5–10", None, "1", "Okutambula", "Sizaangako omwana omufu",
            "Iye", "Orunyankore-Rukiga", "04-May", None, "Ebiseera ebisinga", "1",
            "05-Jul", "Ndi mumativu nnyo", "Iye", "Iye", "Kikulu nnyo ddala",
            "Nsobola", "Mpulira bulungi nnyo"
        ],
        'K-ME-MO-007': [
            "Market cell - kabwohe division", None, None, None, "Emyaka 36–45",
            None, "Kilomita 11–20", None, "Sivaangamu lubuto", "Okutambula",
            "Sizaangako omwana omufu", "Iye", None, "1", "Siniya", "Gyonna",
            "1", "8 nokweyongerayo", "Ndi mumativu", "Nedda", "Iye",
            "Kikulu nnyo ddala", "Simanyi", "Mpulira mu bulungi"
        ],
        'K-ME-013': [
            "Kamwezi A", None, None, None, "Emyaka 18–25", None, "Kilomita 5–10",
            "Oburwadde obusaasana", "Sivaangamu lubuto", "Okutambula",
            "Sizaangako omwana omufu", "Iye", "Orunyankore-Rukiga", "02-Mar",
            "Siniya", "Tekibangako", "02-Mar", "8 nokweyongerayo", "Ndayawo",
            "Nedda", "Nedda", "Kikulu nnyo ddala", "Tekisoboka", "Mpulira bulungi nnyo"
        ],
        'K-ME-MO-010': [
            "Nyakashabya", None, None, None, "Emyaka 18–25", None, "Kilomita 11–20",
            None, "Sivaangamu lubuto", "Okutambula", "Sizaangako omwana omufu",
            "Iye", "Orunyankore-Rukiga", "1", "Essomo lya pulayimale", "Gyonna",
            None, "05-Jul", "Ndayawo", "Nedda", "Nedda", "Kikulu nnyo", "Tekisoboka",
            "Mpulira bulungi nnyo"
        ],
        'K-ME-MO-009': [
            "Rushoroza", None, None, None, "Emyaka 26–35", None, "Kilomita 11–20",
            "Okuleeat omusaayi", "02-Apr", "Motoka zabantu bonna", "Sizaangako omwana omufu",
            "Iye", "Orunyankore-Rukiga,Oruzungu", "02-Mar", "Essomo lya wagulu (ettendekelo/yunivasite)",
            "Gyonna", "02-Mar", "05-Jul", "Ndi mumativu", "Nedda", "Iye",
            "Kikulu nnyo ddala", "Simanyi", "Mpulira mu bulungi"
        ],
        'K-ME-MO-008': [
            "Nyakashambya", None, None, None, "Emyaka 26–35", None, "Kilomita 11–20",
            None, "Sivaangamu lubuto", "Motoka zabantu bonna", "Sizaangako omwana omufu",
            "Iye", "Oluganda,Orunyankore-Rukiga,Oruzungu", "02-Mar",
            "Essomo lya wagulu (ettendekelo/yunivasite)", "Ebiseera ebisinga", None,
            "05-Jul", "Ndayawo", "Iye", "Iye", "Kikulu nnyo", "Simanyi",
            "Mpulira bulungi nnyo"
        ],
        'K-ME-MO-017': [
            "Rwakatsi", None, None, None, "Emyaka 26–35", None, "Kilomita 5–10",
            None, "Sivaangamu lubuto", "Motoka eyange", "Sizaangako omwana omufu",
            "Iye", "Orunyankore-Rukiga", "02-Mar", "Siniya", "Ebiseera ebisinga",
            "1", "05-Jul", "Ndi mumativu nnyo", "Nedda", "Nedda", "Kikulu nnyo ddala",
            "Nsobola", "Mpulira bulungi nnyo"
        ],
        'M/AN/19': [
            "Makenke - Mbarara", "Abasawo mu mabaddwaliro,Redio", "Wiki 10",
            None, "Emyaka 26–35", "Tetanus", "Kilomita 5–10", "Ebilala (biwaandiike)",
            "Sivaangamu lubuto", "Motoka zabantu bonna", "02-Apr", "Iye",
            "Oluganda,Orunyankore-Rukiga", "04-May", "Essomo lya pulayimale",
            "Ebiseera ebisinga", "02-Mar", "05-Jul", "Sili mumativu", "Nedda",
            "Iye", "Kikulu nnyo", "Nsobola", "Mpulira mu bulungi"
        ],
    }
    
    # ============================================================
    # Category 3: Runyankole-Rukiga
    # ============================================================
    about_runyankole_rukiga = [
        "Nyaburawe taho ekyaro kyaawe ……………………",
        "Nogyenda orugyendo rurikwinganaki waba nooza kutunga obuheereza bw’okukyebeza enda abakaikuru/abagurusi abatakazaire?",
        "Ni mirundi engahi ei orikuhurira ngu amakuru agu orikutunga kuruga omu baheereza b‘eby‘amagara nigetegyerezibwa?",
        "Obwegyese bwawe obw’ahaiguru nibuha?",
        "Ni bwire ki obu watungire okugyemesibwa?",
        "Oine emyaka engahi?",
        "Ni rurimi ki oru orikukira kugamba omuka?",
        "Wahisa emirundi engahe orikutwara enda?",
        "Nokunda kutunga amakuru g'eby'amagara omu rurimi ki?",
        "Ozaire abaana bangahi?",
        "Ku kiraabe kiri kityo, toorana ekika (ebika) by‘obuzibu .",
        "N’emirundi engahi ei orugirwe enda y’omwana ofiire.",
        "Ku kiraabe kiri niko kiri, ni omubazi gw’okugyema ndwaraki ogu watungire?",
        "Niki ekikuru ekirikukuremesa kuza omu irwariro kukyebeza enda?",
        "Okagyemesibwa omu bwire bu wabiire oine enda?",
        "Nokira kukozesa ntambura ki waba nooza ahairwariri kuyebeza enda?",
        "Nokunda ngu otungye amakuru gakwatirine n’enda/okuzaara omumuringo ki?",
        "Ni mirundi engahi ei otaayaayiire eirwariro kukyebeza enda?",
        "Obwahati amakuru amaingi agarikukwata aha nda noogatunga kuruga nkahi?",
        "Okagira obuzibu bwona omubwire eine enda?",
        "Noobaasa kukoresa ota puroguraamu y’aha simu erikukuha amakuru agarikukwata aha kuzaara omu rurimi rwawe?",
        "Nohurira oshemereirwe/otiine kwerarikirira kugaaniira aha nshonga eziine akakwate n’okutwara enda n‘abashaho b‘abashaija?",
        "Orakozeiseho app y'aha simu nari aha Intaneeti kutunga amakuru agarikukwata aha nda?",
        "Warakozeiseho app y’esimu kutunga amakuru gakwatirine n’enda/okuzaara?",
        "Oine okumarwa kurikwinganaki n’oburugo bw’amakuru oburiho hati agarikubona garikukwata aha nshonga z'okutwara enda?",
        "Kiine kakuro ki ahariiwe amakuru agarikukwata aha nda kuheebwa omu rurimi rwawe rw'enzaarwa?"
    ]
    
    answers_runyankole_rukiga = {
        'MBR-AD-001': [
            "Katete", "Ahansi ya kilomita 5", "Obwire obwingi",
            "Obwegyese bwa yunivasite/eitendekyero", "Aha wi ki6", "Emyaka 18–25",
            "Orunyankore-Rukiga", "02-Mar", "Orunyankore-Rukiga", "02-Mar",
            "Ebindi (nyaburawe bihandikye)", "Ngaaha", "Tetanas", "Ebeyi/esente",
            "Eego", "Entambura y'abantu boona (taksi)", "Omukugambirwa",
            "Emirundi 5-7", "TV", "Eego", "Nikibaasika munonga",
            "Nimba nyikaikiine munonga", "Eego", "Eego", "Nyine okumarwa kwingi munonga",
            "Nikikuru n’okukirayo"
        ],
        'MBR-AD-002': [
            "Kamukuzi", "Kilomita 5–10", "Obwire obwingi", "Obwegyese bwa siniya",
            None, "Emyaka 18–25", "Orunyankore-Rukiga", "02-Mar", "Orunyankore-Rukiga",
            "02-Mar", None, "Ngaaha", None, "Orugyendo/entambura", "Ngaaha",
            "Entambura y'abantu boona (taksi)", "Omukugambirwa", "Emirundi 2 -4",
            "Abawereza b’okukyalo (VHTs)", None, "Nikibaasika munonga",
            "Nimba ntaine kwikikana kwingi", "Eego", "Ngaaha", "Nyine okumarwa",
            "Nikikuru munonga"
        ],
        'MBR-AD-003': [
            "KIBINGO", "Kilomita 5–10", "Obwire obwingi", "Obwegyese bwa siniya",
            "Aha wi ki6", "Emyaka 18–25", "Orunyankore-Rukiga", "02-Mar",
            "Orunyankore-Rukiga", "02-Mar", None, "Ngaaha", "Oburwaire bwa Hepatitis B",
            "Orugyendo/entambura", "Eego", "Entambura y'abantu boona (taksi)",
            "Omukugambirwa", "Emirundi 2 -4", "Abawereza b’okukyalo (VHTs)", "Eego",
            "Nikibaasika munonga", "Nimba nyikaikiine munonga", "Eego", "Eego",
            "Nyine okumarwa kwingi munonga", "Nikikuru munonga"
        ],
        'MBR-KB-004': [
            "KABEREBERE", "Kilomita 11–20", "Obwire obwingi", "Obwegyese bwa siniya",
            "Aha wiki 14", "Emyaka 26–35", "Orunyankore-Rukiga", "02-Mar",
            "Orunyankore-Rukiga", "02-Mar", None, "Ngaaha", "Tetanas",
            "Obutamanya rurimi", "Eego", "Entambura y'abantu boona (taksi)",
            "Omukugambirwa", "Emirundi 5-7", "Abasawo mu mabaddwaliro", "Ngaaha",
            "Nibaasa kubaskia", "Nimba ntaine kwikikana kwingi", "Eego", "Eego",
            "Tindikumanya gye", "Nikikuru"
        ],
        'MBR-AD-004': [
            "KATETE", "Ahansi ya kilomita 5", "Obwire obwingi", "Obwegyese bwa siniya",
            "Aha wi ki6", "Emyaka 26–35", "Orunyankore-Rukiga", "04-May",
            "Orunyankore-Rukiga", "02-Mar", None, "Ngaaha", "Tetanas",
            "Orugyendo/entambura", "Eego", "Entambura y'abantu boona (taksi)",
            "Omukugambirwa", "Emirundi 5-7", "Abasawo mu mabaddwaliro", "Ngaaha",
            "Nikibaasika", "Nimba nyikaikiine munonga", "Eego", "Eego",
            "Nyine okumarwa", "Nikikuru munonga"
        ],
        'MBR-AD-005': [
            "RWEMIGINA", "Kilomita 5–10", "Obwire obwingi", "Obwegyese bwa purayimare",
            "Aha wiki 14", "Emyaka 26–35", "Orunyankore-Rukiga", "02-Mar",
            "Orunyankore-Rukiga", "1", None, "Ngaaha", "Tetanas", "Orugyendo/entambura",
            "Eego", "Entambura y'abantu boona (taksi)", "Omukugambirwa", "Emirundi 2 -4",
            "Abasawo mu mabaddwaliro", "Ngaaha", "Nikibaasika", "Nimba nyikaikiine munonga",
            "Ngaaha", "Ngaaha", "Nyine okumarwa kwingi munonga", "Nikikuru munonga"
        ],
        'K-ME-MO-002': [
            "Kabwohe Town council", "Kilomita 5–10", "Buriijo", "Obwegyese bwa purayimare",
            "Aha wiki 10", "Emyaka 26–35", "Orunyankore-Rukiga", "04-May",
            "Oluganda", "04-May", "Okujwa eshagama,Pureesha y'ahaiguru,Ebisha ahanyima obwire butakahikire,Oburwaire",
            "Ngaaha", "Tetanas", "Ebeyi/esente", "Eego", "Okutambura",
            "Omukugambirwa", "Emirundi 5-7", "Abawereza b’okukyalo (VHTs)", "Eego",
            "Nibaasa kubaskia", "Nimba nyikaikinehogye,Nimba nyikaikiine munonga",
            "Ngaaha", "Ngaaha", "Tindikumanya gye", "Nikikuru"
        ],
        'MBR-KB-005': [
            "RUBINDI", "Ahaiguru ya kilomita 20", "Obwire obwingi", "Obwegyese bwa purayimare",
            "Aha wiki 14", "Emyaka 26–35", "Orunyankore-Rukiga", "02-Mar",
            "Orunyankore-Rukiga", "02-Mar", None, "Ngaaha", "Tetanas",
            "Orugyendo/entambura", "Eego", "Entambura y'abantu boona (taksi)",
            "Omukugambirwa", "Emirundi 5-7", "Abasawo mu mabaddwaliro", "Ngaaha",
            "Nibaasa kubaskia", "Nimba ntaine kwikikana kwingi", "Ngaaha", "Ngaaha",
            "Nyine okumarwa", "Nikikuru"
        ],
        'MBR-KB-012': [
            "RUBEHO", "Kilomita 5–10", "Obwire obwingi", "Obwegyese bwa yunivasite/eitendekyero",
            None, "Emyaka 26–35", "Orunyankore-Rukiga", "1", "Orushwahili", None,
            None, "Ngaaha", None, None, "Ngaaha", "Emotoka yangye Private vehicle",
            "Omubihandiiko", "Omurundi 1", "Abasawo mu mabaddwaliro", "Ngaaha",
            "Nikibaasika", "Nimba nyikaikiine munonga", "Eego", "Ngaaha",
            "Nyine okumarwa", "Nikikuru n’okukirayo"
        ],
        'MBR-KB-017': [
            "KAKOBA", "Kilomita 5–10", "Buriijo", "Tinyine bwegyese", "Aha wiki 14",
            "Emyaka 36–45", "Orunyankore-Rukiga", "04-May", "Orunyankore-Rukiga",
            "04-May", "Okurugwamu enda/ kuzaara omwana afiire", "1", "Tetanas",
            "Ebeyi/esente", "Eego", "Entambura y'abantu boona (taksi)", "Omukugambirwa",
            "Emirundi 5-7", "Abasawo mu mabaddwaliro", "Eego", "Nibaasa kubaskia",
            "Nimba nyikaikiine munonga", "Ngaaha", "Ngaaha", "Nyine okumarwa kwingi munonga",
            "Nikikuru munonga"
        ],
        'MBR-KB-019': [
            "KATETE-KAMPALA", "Kilomita 5–10", "Obwire obwingi", "Obwegyese bwa siniya",
            "Aha wiki 14", "Emyaka 26–35", "Orunyankore-Rukiga", "1",
            "Orunyankore-Rukiga", None, None, "Ngaaha", "Tetanas", "Ebeyi/esente",
            "Eego", "Entambura y'abantu boona (taksi)", "Omukugambirwa", "Emirundi 2 -4",
            "Abasawo mu mabaddwaliro", "Ngaaha", "Nikibaasika", "Nimba nyikaikiine munonga",
            "Ngaaha", "Ngaaha", "Nyine okumarwa kwingi munonga", "Nikikuru n’okukirayo"
        ],
        'MBR-NP-022': [
            "KORANORYA", "Ahansi ya kilomita 5", "Obwire obwingi", "Obwegyese bwa siniya",
            None, "Emyaka 18–25", "Orunyankore-Rukiga", "02-Mar", "Orunyankore-Rukiga",
            "1", None, "Ngaaha", "Tetanas", "Okubaho/okutabaho kw'abaheereza b'eby'amagara",
            "Eego", "Entambura y'abantu boona (taksi)", "Omukugambirwa", "Emirundi 2 -4",
            "Abasawo mu mabaddwaliro", "Ngaaha", "Nikibaasika munonga",
            "Tinyine rubaju", "Ngaaha", "Ngaaha", "Nyine okumarwa",
            "Nikikuru n’okukirayo"
        ],
        'MBR-NP-021': [
            "NKOKONJERU", "Ahansi ya kilomita 5", "Buriijo", "Obwegyese bwa siniya",
            None, "Emyaka 18–25", "Orunyankore-Rukiga", "1", "Orunyankore-Rukiga",
            "1", None, "Ngaaha", "Tetanas", "Ekindi (nyaburawe kigambe)",
            "Eego", "Entambura y'abantu boona (taksi)", "Omukugambirwa", "Omurundi 1",
            "Abasawo mu mabaddwaliro", "Ngaaha", "Nikibaasika", "Nimba ntaine kwikikana kwingi",
            "Ngaaha", "Ngaaha", "Nyine okumarwa", "Nikikuru n’okukirayo"
        ],
        'MBR-NP-020': [
            "KATETE MBARARA", "Ahansi ya kilomita 5", "Buriijo",
            "Obwegyese bwa yunivasite/eitendekyero", "Ahakuzaara,Aha wi ki6",
            "Emyaka 26–35", "Oluganda", "02-Mar", "Oluganda", "02-Mar",
            "Oburwaire", "Ngaaha", "Tetanas", "Ebeyi/esente", "Eego",
            "Entambura y'abantu boona (taksi)", "Omukugambirwa", "Emirundi 5-7",
            "Abasawo mu mabaddwaliro", "Eego", "Nikibaasika munonga",
            "Nimba nyikaikiine munonga", "Eego", "Ngaaha", "Nyine okumarwa kwingi munonga",
            "Nikikuru n’okukirayo"
        ],
        'MBR-NP-010': [
            "RUBINDI", "Ahansi ya kilomita 5", "Buriijo",
            "Obwegyese bwa yunivasite/eitendekyero", None, "Emyaka 18–25",
            "Orunyankore-Rukiga", "1", "Orunyankore-Rukiga", None, None, "Ngaaha",
            "Tetanas", "Okubaho/okutabaho kw'abaheereza b'eby'amagara", "Eego",
            "Okutambura", None, "Emirundi 5-7", "Abafamire n’emikwano", "Ngaaha",
            "Nikibaasika munonga", "Nimba nyikaikiine munonga", "Eego", "Eego",
            "Nyine okumarwa", "Nikikuru n’okukirayo"
        ],
        'MBR-NP-006': [
            "RUBIRI", "Ahansi ya kilomita 5", "Buriijo", "Obwegyese bwa siniya",
            None, "Emyaka 36–45", "Oluganda", "02-Mar", "Oluganda", "02-Mar",
            None, "Ngaaha", "Tetanas", "Okubaho/okutabaho kw'abaheereza b'eby'amagara",
            "Eego", "Entambura y'abantu boona (taksi)", "Omukugambirwa", "Emirundi 2 -4",
            "Abasawo mu mabaddwaliro", "Ngaaha", "Nikibaasika", "Nimba ntaine kwikikana kwingi",
            "Ngaaha", "Ngaaha", None, "Nikikuru n’okukirayo"
        ],
        'MBR-NA-016': [
            "MASHERUKA", "Ahansi ya kilomita 5", "Buriijo", "Obwegyese bwa purayimare",
            "Aha myezi 6", "Emyaka 26–35", None, "02-Mar", "Orunyankore-Rukiga",
            "02-Mar", None, "Ngaaha", "Tetanas", "Ebeyi/esente", "Eego",
            "Okutambura", "Omukugambirwa", "Emirundi 5-7", "Abawereza b’okukyalo (VHTs)",
            "Ngaaha", "Nikibaasika", "Nimba nyikaikiine munonga", "Ngaaha", "Eego",
            "Nyine okumarwa", "Nikikuru munonga"
        ],
        'MBR-NA-017': [
            "KIHANDA-SHEEMA", "Ahansi ya kilomita 5", "Obwire obwingi",
            "Obwegyese bwa purayimare", None, "Emyaka 26–35", "Orunyankore-Rukiga",
            "04-May", "Orunyankore-Rukiga", "02-Mar", "Oburwaire", "Ngaaha",
            None, "Ebeyi/esente", "Eego", "Okutambura", "Omukugambirwa",
            "Emirundi 5-7", "Redio", "Eego", "Nibaasa kubaskia", "Nimba ntaine kwikikana kwingi",
            "Eego", "Eego", "Tinyine kumarwa", "Nikikuru munonga"
        ],
        'MBR-NA-020': [
            "PRISONS", "Ahansi ya kilomita 5", "Buriijo", "Obwegyese bwa siniya",
            "Aha wi ki6", "Emyaka 26–35", "Orundi (Ruhandiike nyaburawe)", "04-May",
            "Orujungu", "1", "Pureesha y'ahaiguru", "2 – 4", "Tetanas",
            "Ekindi (nyaburawe kigambe)", "Eego", "Okutambura", "Omukugambirwa",
            "Emirundi 5-7", "TV", "Eego", "Nikibaasika munonga", "Nimba nyikaikiine munonga",
            "Eego", "Eego", "Nyine okumarwa kwingi munonga", "Nikikuru n’okukirayo"
        ],
        'MBR-NA-021': [
            "RUHARO", "Kilomita 5–10", "Obwire obwingi", "Obwegyese bwa purayimare",
            "Ahakuzaara", "Emyaka 18–25", "Orunyankore-Rukiga", "1", "Orunyankore-Rukiga",
            "02-Mar", None, "Ngaaha", "Tetanas", "Ekindi (nyaburawe kigambe)",
            "Eego", "Ezindi (nyaburawe zihandiikye)", "Omukugambirwa", "Emirundi 5-7",
            "Abasawo mu mabaddwaliro", "Ngaaha", "Tikirikubaasika", "Nimba ntaine kwikikana kwingi",
            "Eego", "Ngaaha", "Nyine okumarwa", "Nikikuru"
        ],
        'MBR-NA-023': [
            "RUTI", "Ahansi ya kilomita 5", "Obwire obwingi", "Obwegyese bwa siniya",
            "Aha wiki 14", "Emyaka 36–45", "Oluganda,Orunyankore-Rukiga", "04-May",
            "Oluganda,Orunyankore-Rukiga", "04-May", "Ebindi (nyaburawe bihandikye)",
            "Ngaaha", "Tetanas", "Ebeyi/esente", "Ngaaha", "Okutambura",
            "Omukugambirwa", "Emirundi 2 -4", "Redio", "Eego", "Nibaasa kubaskia",
            "Nimba ntaine kwikikana kwingi", "Ngaaha", "Ngaaha", "Tindikumanya gye",
            "Nikikuru munonga"
        ],
        'K-ME-MO-005': [
            "KACORONGOTO", "Ahansi ya kilomita 5", "Obwire obwingi",
            "Obwegyese bwa purayimare", "Ahakuzaara,Aha wi ki6", "Emyaka 18–25",
            "Orunyankore-Rukiga", "02-Mar", "Orunyankore-Rukiga", "02-Mar",
            None, "Ngaaha", "Tetanas", "Orugyendo/entambura", "Eego",
            "Emotoka yangye Private vehicle", "Omukugambirwa", "Emirundi 2 -4",
            "Redio", "Ngaaha", "Nikibaasika", "Nimba nyikaikinehogye", "Ngaaha",
            "Ngaaha", "Tindikumanya gye", "Nikikuru n’okukirayo"
        ],
        'K-ME-MO-004': [
            "Buteganio", "Ahaiguru ya kilomita 20", "Obwire obwingi", "Tinyine bwegyese",
            "Aha wiki 14", "Emyaka 18–25", "Orunyankore-Rukiga", "1",
            "Orunyankore-Rukiga", "1", "Oburwaire", "Ngaaha", "Tetanas", "Ebeyi/esente",
            "Eego", "Entambura y'abantu boona (taksi)", "Omukugambirwa", "Emirundi 5-7",
            "Redio", "Eego", "Nikibaasika", "Nimba nyikaikiine munonga", "Ngaaha",
            "Ngaaha", "Nyine okumarwa", "Nikikuru n’okukirayo"
        ],
        'K-ME-MO-016': [
            "Kabwohe Town A", "Kilomita 5–10", "Obwire bumwe/obumwe na bumwe",
            "Obwegyese bwa yunivasite/eitendekyero", "Aha wiki 10", "Emyaka 26–35",
            "Orunyankore-Rukiga", "02-Mar", "Orunyankore-Rukiga", "1", None,
            "Ngaaha", "Tetanas", "Orugyendo/entambura", "Eego", "Okutambura",
            "Omukugambirwa", "Emirundi 8 nokurengamu", "Abawereza b’okukyalo (VHTs)",
            "Ngaaha", "Nikibaasika munonga", "Nimba nyikaikiine munonga", "Eego",
            "Eego", "Nyine okumarwa kwingi munonga", "Nikikuru n’okukirayo"
        ],
        'K-ME-MO-014': [
            "Ishekye", "Ahansi ya kilomita 5", "Obwire bumwe/obumwe na bumwe",
            "Obwegyese bwa purayimare", "Aha myezi 6", "Emyaka 18–25",
            "Orunyankore-Rukiga", "02-Mar", "Orunyankore-Rukiga", "1", None,
            "Ngaaha", "Tetanas", "Orugyendo/entambura", "Eego", "Okutambura",
            "Omukugambirwa", "Emirundi 8 nokurengamu", "Abasawo mu mabaddwaliro",
            "Ngaaha", "Nikibaasika munonga", "Nimba nyikaikinehogye", "Eego", "Eego",
            "Nyine okumarwa kwingi munonga", "Nikikuru n’okukirayo"
        ],
        'K-ME-MO-015': [
            "KItoohwa", "Kilomita 5–10", "Obwire obwingi", "Obwegyese bwa purayimare",
            "Aha wi ki6,Aha mwaka 1", "Emyaka 18–25", "Orunyankore-Rukiga", "1",
            "Orunyankore-Rukiga", "1", None, "Ngaaha", "Polio", "Orugyendo/entambura",
            "Eego", "Entambura y'abantu boona (taksi)", "Omukugambirwa", "Emirundi 5-7",
            "Abasawo mu mabaddwaliro", "Ngaaha", "Nikibaasika munonga",
            "Nimba nyikaikiine munonga", "Eego", "Eego", "Nyine okumarwa kwingi munonga",
            "Nikikuru n’okukirayo"
        ],
        'K-ME-MO-001': [
            "Rutooma", "Ahansi ya kilomita 5", "Buriijo",
            "Obwegyese bwa yunivasite/eitendekyero",
            "Ahakuzaara,Aha wi ki6,Aha wiki 10,Aha wiki 14,Aha myezi 6",
            "Emyaka 26–35", "Orunyankore-Rukiga", "1", "Orunyankore-Rukiga", "1",
            "Okujwa eshagama,Ebisha ahanyima obwire butakahikire", "Ngaaha",
            "Tetanas", "Ebeyi/esente", "Eego", "Entambura y'abantu boona (taksi)",
            "Omukugambirwa", "Emirundi 8 nokurengamu", "Abawereza b’okukyalo (VHTs)",
            "Eego", "Nibaasa kubaskia", "Nimba nyikaikiine munonga", "Ngaaha",
            "Ngaaha", "Nyine okumarwa kwingi munonga", "Nikikuru n’okukirayo"
        ],
        'K-ME-MO-003': [
            "Kemikyera cell", "Ahansi ya kilomita 5", "Obwire bumwe/obumwe na bumwe",
            "Obwegyese bwa yunivasite/eitendekyero", "Aha wi ki6", "Emyaka 36–45",
            "Orunyankore-Rukiga", "04-May", "Orunyankore-Rukiga", "02-Mar",
            "Shukari y'omunda/shukari y’obwire bwenda,Okurugwamu enda/ kuzaara omwana afiire",
            "1", "Tetanas", "Ebeyi/esente", "Eego", "Ezindi (nyaburawe zihandiikye)",
            "Omukugambirwa", "Emirundi 8 nokurengamu", "Abasawo mu mabaddwaliro",
            "Eego", "Nibaasa kubaskia", "Nimba nyikaikiine munonga", "Ngaaha",
            "Ngaaha", "Nyine okumarwa kwingi munonga", "Nikikuru n’okukirayo"
        ],
        'K-JK-MO-020': [
            "Kemichera", "Ahansi ya kilomita 5", "Buriijo", "Obwegyese bwa siniya",
            None, "Emyaka 18–25", "Orunyankore-Rukiga,Orundi (Ruhandiike nyaburawe)",
            "1", "Orunyankore-Rukiga", None, "Oburwaire", "Ngaaha", None,
            "Ebeyi/esente", "Ngaaha", "Entambura y'abantu boona (taksi)",
            "Omukugambirwa", "Omurundi 1", "Abafamire n’emikwano", "Eego",
            "Nikibaasika munonga", "Nimba nyikaikiine munonga", "Eego", "Eego",
            "Nyine okumarwa kwingi munonga", "Nikibaasa kuba kiri kikuru"
        ],
        'K-JK-MO-012': [
            "Rutooma Cell", "Ahansi ya kilomita 5", "Obwire bumwe/obumwe na bumwe",
            "Obwegyese bwa siniya", "Aha wiki 14", "Emyaka 26–35",
            "Orunyankore-Rukiga", "02-Mar", "Orunyankore-Rukiga", "02-Mar",
            None, "1", "Tetanas", "Ebeyi/esente", "Eego", "Entambura y'abantu boona (taksi)",
            "Omukugambirwa", "Emirundi 8 nokurengamu", "Abasawo mu mabaddwaliro",
            "Ngaaha", "Nikibaasika munonga", "Tinyine rubaju", "Eego", "Eego",
            "Nyine okumarwa", "Nikikuru munonga"
        ],
    }
    
    # ============================================================
    # Returning the data as a dictionary
    # ============================================================
    self.data = {
        "about_english": {
            "questions": about_english,
            "answers": answers_english
        },
        "about_luganda": {
            "questions": about_luganda,
            "answers": answers_luganda
        },
        "about_runyankole_rukiga": {
            "questions": about_runyankole_rukiga,
            "answers": answers_runyankole_rukiga
        }
    }
    return self.data