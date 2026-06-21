"""
Maternal Multilingual Dataset - East African Maternal Health QA
8 Categories, Multiple Questions, 4 Languages (English, Swahili, Luganda, Runyankore)
"""

from typing import Dict, List


class MaternalMultilingualDataset:
    """Multilingual maternal health dataset"""
    
    def __init__(self):
        self.data = None
    
    def load_your_data(self) -> Dict:
        """Load the maternal multilingual dataset"""
        
        # ============================================================
        # Category 1: About healthcare access
        # ============================================================
        about_healthcare_access = [
            "Have you had any previous pregnancies, and were there any complications?",
            "Do you use any local herbs or traditional medicines?",
            "What type of daily activities do you perform, and do you find them physically demanding?"
        ]
        
        answers_about_healthcare_access = {
            'English': [
                "Yes, I lost the baby before giving birth.",
                "I sometimes use ginger, but I prefer consulting a doctor.",
                "My days mostly involve digging. It's quite physically demanding."
            ],
            'Swahili': [
                "Ndiyo, nilipoteza mtoto kabla ya kujifungua.",
                "Wakati mwingine mimi hutumia tangawizi, lakini napendelea kushauriana na daktari.",
                "Siku zangu mara nyingi huhusisha kulima. Kunahitaji sana nguvu za kimwili."
            ],
            'Luganda': [
                "Yee, omwana namufiirwa nga sinnazaala.",
                "Oluusi nkozesa entangawuzzi naye nsinga kwagala kwebuuza ku musawo.",
                "Enaku zange ezisinga obungi nzimalira mu kulima. Kyetaagisa amaanyi mangi mu mubiri."
            ],
            'Runyankore': [
                "Eego, nkaferwa omwana ntakazaire.",
                "Obumwe n'obumwe ninkoresa entangawuuzi, kwonka ninkunda kwebuuza aha mushaho.",
                "Ebiro byangye nibikira kuba birimu okuhinga. Nikinkoresa ebisasaizi."
            ]
        }
        
        # ============================================================
        # Category 2: About medical history and lifestyle
        # ============================================================
        about_medical_history_lifestyle = [
            "How do you manage stress, and do you have anyone you can talk to?",
            "Have you experienced any unusual pain or discomfort recently?",
            "How are you managing the common pregnancy symptoms?"
        ]
        
        answers_about_medical_history_lifestyle = {
            'English': [
                "I talk to my family members when I feel stressed.",
                "Thankfully, I haven't really been experiencing those common pregnancy symptoms.",
                "If I have issues, I rest or seek medical help."
            ],
            'Swahili': [
                "Ninazungumza na wanafamilia yangu ninapohisi msongo wa mawazo.",
                "Kwa bahati nzuri, sijapata dalili hizo za kawaida za ujauzito.",
                "Ikiwa nina matatizo, mimi hupumzika tu, au ninatafuta usaidizi wa matibabu."
            ],
            'Luganda': [
                "Njogera n'ab'omu maka gange bwe mba ne situleesi.",
                "Ekirungi, mu butuufu sifunangako bubonero obwo obwa bulijjo obw'olubuto.",
                "Bwe mba nnina ensonga, mpummula buwummuzi, oba nnoonya obuyambi bw'abasawo."
            ],
            'Runyankore': [
                "Ningamba n'abantu b'omuka yangye ku ndikwehurira nyine oburemeezi.",
                "Ekirungi, tinkatunga obubonero bw'okutwara enda oburikukira kureebeka.",
                "Naaba nyine oburemeezi, nimpumuraho, nainga ninsherura obuhwezi bw'abashaho."
            ]
        }
        
        # ============================================================
        # Category 3: About mental health and support
        # ============================================================
        about_mental_health_support = [
            "How do you feel about becoming a mother?",
            "Do you have any fears about giving birth?",
            "What support do you have at home?"
        ]
        
        answers_about_mental_health_support = {
            'English': [
                "I feel very happy about becoming a mother.",
                "I do feel some fear about the pain of giving birth.",
                "I have good support at home from my husband and mother."
            ],
            'Swahili': [
                "Ninajisikia furaha sana kuwa mama.",
                "Ninahisi hofu fulani juu ya uchungu wa kuzaa.",
                "Nina usaidizi mzuri nyumbani kutoka kwa mume wangu na mama yangu."
            ],
            'Luganda': [
                "Mpulira essanyu lingi nnyo okubeera maama.",
                "Mpulira okutya okutonotono ku bulumi bw'okuzaala.",
                "Nina obuyambi obulungi awaka okuva eri omwami wange ne maama."
            ],
            'Runyankore': [
                "Nimpurira okushemererwa ahabw'okuba maama.",
                "Nyine okutiina okukye ahabw'obusaasi bw'okuzaara.",
                "Nyine obuhwezi omuka kuruga aha mushaija wangye na maama."
            ]
        }
        
        # ============================================================
        # Category 4: About recovery and baby care
        # ============================================================
        about_recovery_baby_care = [
            "How are you recovering after childbirth?",
            "Are you able to breastfeed your baby?",
            "Do you have any concerns about your baby's health?"
        ]
        
        answers_about_recovery_baby_care = {
            'English': [
                "I am recovering well with support from my family.",
                "Yes, I am able to breastfeed my baby regularly.",
                "My baby seems healthy and I don't have major concerns."
            ],
            'Swahili': [
                "Ninapata nafuu vizuri kwa msaada wa familia yangu.",
                "Ndiyo, ninaweza kumnyonyesha mtoto wangu mara kwa mara.",
                "Mtoto wangu anaonekana mwenye afya na sina wasiwasi mkubwa."
            ],
            'Luganda': [
                "Ntereera bulungi nga nyambibwako ab'omu maka gange.",
                "Yee, nsobola okuyonsa omwana wange buli kiseera.",
                "Omwana wange alabika mulamu era sirina buzibu bunene."
            ],
            'Runyankore': [
                "Nintwaza kukira kurungi n'obuhwezi bw'omuka yangye.",
                "Eego, nimbaasa kwonsa omwana wangye buri kaire.",
                "Omwana wangye naareebeka nk'aine amagara marungi kandi tinyine buremeezi bunene."
            ]
        }
        
        # ============================================================
        # Category 5: About recovery and health
        # ============================================================
        about_recovery_health = [
            "Have you had any complications after delivery?",
            "Are you taking any medication for recovery?",
            "How is your overall health since giving birth?"
        ]
        
        answers_about_recovery_health = {
            'English': [
                "No, I haven't had any complications after delivery.",
                "Yes, I am taking the medication prescribed by the doctor.",
                "My overall health is improving day by day."
            ],
            'Swahili': [
                "Hapana, sijapata matatizo yoyote baada ya kujifungua.",
                "Ndiyo, ninatumia dawa nilizoagizwa na daktari.",
                "Afya yangu kwa ujumla inazidi kuwa bora siku hadi siku."
            ],
            'Luganda': [
                "Nedda, sifunye buzibu bwonna nga mmaze okuzaala.",
                "Yee, mmira eddagala eriragiddwa omusawo.",
                "Okutwalira awamu obulamu bwange bweyongera okuba obulungi buli lunaku."
            ],
            'Runyankore': [
                "Ngaaha, tinkatungire buremeezi bwona bwanyima y'okuzaara.",
                "Eego, ninkoresa omubazi ogu omushaho yaangambiire.",
                "Amagara gangye okutwarira hamwe, nirikweyongyera gurungi buri eizooba."
            ]
        }
        
        # ============================================================
        # Category 6: About symptoms and concerns
        # ============================================================
        about_symptoms_concerns = [
            "What symptoms are you experiencing?",
            "Have you had any bleeding after delivery?",
            "Do you have any pain that concerns you?"
        ]
        
        answers_about_symptoms_concerns = {
            'English': [
                "I have some mild discomfort but nothing severe.",
                "I had normal bleeding that is decreasing each day.",
                "I have some lower abdominal pain that is manageable."
            ],
            'Swahili': [
                "Nina usumbufu kidogo lakini si kali.",
                "Nilikuwa na kutokwa na damu kwa kawaida ambayo inapungua kila siku.",
                "Nina maumivu kidogo ya chini ya tumbo ambayo yanaweza kudhibitiwa."
            ],
            'Luganda': [
                "Nfuna obutabeera bulungi katono naye si kya maanyi.",
                "Nafulumya omusaayi ogwa bulijjo ogukendeera buli lunaku.",
                "Nfuna obulumi obutonotono mu lubuto olwa wansi obukakanyizibwa."
            ],
            'Runyankore': [
                "Nyine oburemeezi bukye kwonka tiburi bwingi.",
                "Nkajwa eshagama ya buriijo erikukyenda buli eizooba.",
                "Nyine obusaasi bukye omu nda-nto oburikyendeezibwa."
            ]
        }
        
        # ============================================================
        # Category 7: About traditional and cultural practices
        # ============================================================
        about_traditional_cultural_practices = [
            "Do you use any traditional remedies?",
            "What cultural practices do you follow after childbirth?",
            "Do you combine traditional and modern medicine?"
        ]
        
        answers_about_traditional_cultural_practices = {
            'English': [
                "Sometimes I use herbal baths for recovery.",
                "We have a practice of staying indoors for a week after birth.",
                "I use both traditional herbs and hospital medicine."
            ],
            'Swahili': [
                "Wakati mwingine mimi hutumia mitishamba kuoga kwa ajili ya kupona.",
                "Tuna desturi ya kukaa ndani ya nyumba kwa wiki moja baada ya kujifungua.",
                "Mimi hutumia mitishamba na dawa za hospitali pamoja."
            ],
            'Luganda': [
                "Oluusi nkozesa eddagala ly'ekinnansi okunaaba okuwona.",
                "Tulina enkola y'okusigala mu nnyumba okumala wiiki emu nga tumaze okuzaala.",
                "Nkozesa eddagala ly'ekinnansi n'ery'eddwaaliro bombi."
            ],
            'Runyankore': [
                "Obumwe n'obumwe ninkoresa emibazi y'obuhangwa okunaaba kwenda kukira.",
                "Tunamu omucwe gw'okuguma omuka kumara wiki emwe bwanyima y'okuzaara.",
                "Ninkoresa emibazi y'obuhangwa hamwe n'ey'eirwariro zombi."
            ]
        }
        
        # ============================================================
        # Category 8: Community and cultural considerations
        # ============================================================
        about_community_cultural_considerations = [
            "What are common beliefs about pregnancy in your community?",
            "Do community elders influence your health decisions?",
            "How do you balance cultural practices with medical advice?"
        ]
        
        answers_about_community_cultural_considerations = {
            'English': [
                "Many believe in using herbs to ensure a safe delivery.",
                "Yes, elders are highly respected and their advice is followed.",
                "I try to respect cultural practices while following doctor's orders."
            ],
            'Swahili': [
                "Wengi wanaamini katika kutumia mitishamba kuhakikisha kujifungua salama.",
                "Ndiyo, wazee wanaheshimiwa sana na ushauri wao unafuatwa.",
                "Ninajaribu kuheshimu mila za kitamaduni huku nikifuata maagizo ya daktari."
            ],
            'Luganda': [
                "Bangi bakkiririza mu kukozesa eddagala ly'ekinnansi okusobola okuzaala bulungi.",
                "Yee, abakadde bassa ekitiibwa era amagezi gaabwe gagobererwa.",
                "Nfuba okussa ekitiibwa mu nkola z'obuwangwa nga nkola nga bwe ndagiddwa omusawo."
            ],
            'Runyankore': [
                "Abantu baingi nibakiriza ngu okukoresa emibazi y'obuhangwa nikuhwera kuzaara kurungi.",
                "Eego, abakuru bahaibwa ekitiinisa kandi obuhabuzi bwabo nibukuratirwa.",
                "Ningyezaho kuha ekitinisa eby'obuhangwa n'obwo ndikukuratira ebiragiro by'omushaho."
            ]
        }
        
        # ============================================================
        # Returning the data as a dictionary
        # ============================================================
        self.data = {
            "about_healthcare_access": {
                "questions": about_healthcare_access,
                "answers": answers_about_healthcare_access
            },
            "about_medical_history_lifestyle": {
                "questions": about_medical_history_lifestyle,
                "answers": answers_about_medical_history_lifestyle
            },
            "about_mental_health_support": {
                "questions": about_mental_health_support,
                "answers": answers_about_mental_health_support
            },
            "about_recovery_baby_care": {
                "questions": about_recovery_baby_care,
                "answers": answers_about_recovery_baby_care
            },
            "about_recovery_health": {
                "questions": about_recovery_health,
                "answers": answers_about_recovery_health
            },
            "about_symptoms_concerns": {
                "questions": about_symptoms_concerns,
                "answers": answers_about_symptoms_concerns
            },
            "about_traditional_cultural_practices": {
                "questions": about_traditional_cultural_practices,
                "answers": answers_about_traditional_cultural_practices
            },
            "community_cultural_considerations": {
                "questions": about_community_cultural_considerations,
                "answers": answers_about_community_cultural_considerations
            }
        }
        return self.data