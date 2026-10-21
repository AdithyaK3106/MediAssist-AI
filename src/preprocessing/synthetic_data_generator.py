import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Clinically authentic synthetic datasets for the 8 high-prevalence Indian outpatient conditions
SYNTHETIC_CONDITIONS_DATA = {
    "Tension Headache": [
        # Short isolated queries typical of outpatient users
        "headache",
        "headache since 3 days",
        "headache since 2 days",
        "headache since 4 days",
        "headache since yesterday",
        "headache from 3 days",
        "continuous headache",
        "mild headache",
        "dull headache",
        "dull aching headache",
        "headache no fever",
        "headache without fever",
        "only headache since 3 days",
        "just a headache",
        "headache and stress",
        "tight band headache",
        "sar dard",
        "sir dard",
        "sar me dard",
        "3 din se sar dard",
        "halka sar dard",
        "sar me bhari pan",

        # Conversational English & Detailed queries
        "I have a continuous dull aching headache since 3 days.",
        "Headache since 3 days, feels like a tight band wrapped around my forehead.",
        "Constant dull pressure across my temples and back of the head for the past few days.",
        "Mild but constant headache from morning till night, feels heavy.",
        "Tight band-like sensation pressing around my entire skull, no nausea or vomiting.",
        "Dull head pain, tender scalp muscles and tightness in back of the neck.",
        "Aching head pain since yesterday, exacerbated by long hours at my office desk.",
        "Pressure headache on both sides of my forehead, no sensitivity to light.",
        "Moderate dull ache in my head that doesn't throb, just feels like heavy tension.",
        "Continuous pressure in head for 2-3 days, feeling exhausted and stressed.",
        "Mild bilateral headache, feels like someone is squeezing my forehead.",
        "Dull persistent head ache after working late hours, painkillers give only temporary relief.",
        "My head feels uncomfortably tight and heavy since the past 48 hours.",
        "Slight headache and tight neck muscles after prolonged computer work and lack of sleep.",
        "Non-throbbing headache on both sides, worse by evening, no fever or vomiting.",
        
        # Indian English colloquial
        "Having continuous headache since 3 days due to office stress and work pressure.",
        "Head feeling very heavy and tight from last three days, no vomiting or cold.",
        "Facing dull head pain continuously since two days, neck is also feeling stiff.",
        "Too much heavy feeling in head and forehead after late night work.",
        "Mild continuous head ache, not sharp but continuous dull pain since yesterday.",
        "Head is aching constantly without any fever or cold, just tightness around temples.",
        "Getting continuous pain across forehead and back of skull due to lack of proper sleep.",
        "Dull aching pain in whole head since three days, feeling very drained.",
        "Heavy head sensation and pressure over eyebrows since yesterday morning.",
        "Pain on both sides of head like band pressing, no vomiting, just feeling exhausted.",
        
        # Hinglish & Multilingual expressions
        "Sar me bohot heavy pan lag raha hai 3 din se, halka continuous dard hai.",
        "Teen din se sar me lagatar halka dard aur tension jaisa mehsoos ho raha hai.",
        "Sar ke dono taraf daban jaisa dard hai, ulti ya bukhar bilkul nahi hai.",
        "Office ke stress ki wajah se sar dukh raha hai aur gardan me thoda stiffness hai.",
        "Sar ghoom nahi raha par aage forehead par continuous daban aur dard hai 2 din se.",
        "Kal raat se sar me bohot bhari pan hai, neend poori na hone ki wajah se.",
        "Sar me bandha hua jaisa tightness lag raha hai, bukhar ya aankh me dard nahi hai.",
        "Teen din se sar dard ho raha hai, light se problem nahi hai bas continuous daban hai.",
        "Bohot zyada kaam ke baad sar dukh raha hai aur forehead par continuous heavy feel ho raha hai.",
        "Sar me halka halka dard lagatar bana hua hai, ulti ya chakkar nahi hai.",
        "Gardan ke piche aur sar ke aage dono side daban jaisa dard hai.",
        "Subah se sar bhari hai aur temples me pressure jaisa feel ho raha hai.",
        "Sar dard teen din se hai par severe nahi hai, bas dull pain bana rehta hai.",
        "Dimag me tightness aur sar me bohot thakan aur halka dard mehsoos ho raha hai.",
        "Screen par lagatar kaam karne se sar me continuous band pain hai."
    ],

    "Eye Strain / Screen Fatigue": [
        # Short isolated queries
        "eye strain",
        "eyes burning",
        "burning eyes",
        "tired eyes",
        "screen headache",
        "eye fatigue",
        "headache behind eyes from screen",
        "eyes hurt from laptop",
        "aankhon me jalan",
        "aankh me dard",
        "screen time headache",

        # Conversational English & Detailed queries
        "My eyes are burning and I have a headache right behind my eyebrows after using my laptop.",
        "Eyes feel tired, dry, and irritated with a headache across my forehead from screen use.",
        "Frontal headache and burning sensation in both eyes after looking at mobile screen all day.",
        "Blurred vision and eye fatigue after working 10 hours on computer screens.",
        "Heavy forehead ache and difficulty focusing eyes by late evening.",
        "Eyes feeling watery, strained, and stinging after prolonged monitor usage.",
        "Sharp tired feeling in eyes and dull headache around the bridge of the nose.",
        "Eye redness, dry gritty feeling, and frontal head tightness after software coding.",
        "Tired burning eyes, sensitivity to screen brightness, and dull brow ache.",
        "Eyes feel exhausted and heavy, forehead aches when trying to read small text.",
        "Discomfort behind the eyes and dull frontal ache after continuous mobile phone gaming.",
        "Strained eyes, mild double vision when tired, and tension over forehead.",
        "Eyes twitching, burning, and feeling sore after long hours under fluorescent office lights.",
        "Dry itchy eyes and aching forehead, feel much better when closing eyes.",
        "Ache behind eyeballs and frontal head pain by late afternoon at work.",
        
        # Indian English colloquial
        "Eyes burning too much and forehead paining after working whole day on computer.",
        "Screen time is high and getting pain above both eyes with burning sensation.",
        "Continuous mobile viewing causing eye tiredness, watery eyes, and head ache.",
        "Facing dryness in eyes and heavy forehead pain by evening time.",
        "Eyes are paining while looking at laptop screen, feeling gritty and tired.",
        "Getting headache right above eyes whenever working on monitor for more than 2 hours.",
        "Eyes feeling strained and red with dull headache in front portion of head.",
        "Eye tiredness and mild blurring of letters after working long shifts on system.",
        "Too much burning feeling in both eyes, forehead is also aching since yesterday.",
        "Eyes are feeling heavy and strained, difficult to focus on laptop screen.",
        
        # Hinglish & Multilingual expressions
        "Laptop screen dekh dekh ke aankhon me bohot jalan aur sar me aage dard hai.",
        "Aankhein bohot thak gayi hain aur forehead par dard ho raha hai din bhar system pe baithne se.",
        "Screen time zyada hone ki wajah se aankhon me khujli, jalan aur sar dard ho raha hai.",
        "Aankhon ke upar eyebrow ke paas bhari pan aur daban lag rahi hai.",
        "Aankhein laal ho gayi hain aur computer screen dekhne me takleef ho rahi hai.",
        "Phone chalane ke baad aankhon me pani aa raha hai aur sar dukh raha hai.",
        "Aankhon ke piche dard hai aur screen ki brightness se aankh me chubhan lag rahi hai.",
        "Aankh bohot dry aur tired lag rahi hai, band karne par aaram milta hai.",
        "Aankhon me jalan aur forehead par heavy headache hai shaam se.",
        "Pura din coding karne ke baad aankhon me kankari jaisa lag raha hai aur sar dard hai.",
        "Subah se screen dekhne ke baad aankhon me dukhne jaisa dard aur mild blurriness hai.",
        "Aankhon par bohot zor pad raha hai aur aage sar me daban ho rahi hai.",
        "Aankhein jhapakne par jalan aur forehead me continuous thakan mehsoos ho rahi hai.",
        "Mobile dekhne ke baad aankhon me pain aur aage sar me thoda pressure hai.",
        "Aankhein thak kar band ho rahi hain aur sir ke aage wale hisse me dard hai."
    ],

    "Sinusitis": [
        # Short isolated queries
        "sinus headache",
        "sinus pressure",
        "blocked nose and headache",
        "sinus pain",
        "facial headache",
        "heavy forehead and blocked nose",
        "headache when bending",
        "pain behind cheekbones",
        "naak band aur sar dard",
        "sinus",

        # Conversational English & Detailed queries
        "Severe facial pressure around my cheekbones and forehead with a blocked nose.",
        "Frontal headache that throbs and worsens when I bend forward to pick up things.",
        "Nasal congestion, thick yellow nasal discharge, and heaviness behind my eyes.",
        "Face feels congested, tender cheekbones, blocked nasal passages, and dull head pressure.",
        "Pain and fullness in my forehead and temples, nose is completely stuffed up.",
        "Headache centered around my nose bridge, cheeks, and forehead since 4 days.",
        "Post-nasal drip, coughing from mucus dripping down throat, and heavy sinus pressure.",
        "Tender cheeks to touch, stuffed nose, loss of smell, and pressure behind eyes.",
        "Woke up with heavy forehead, congested nose, and ear fullness following a cold.",
        "Dull throbbing pain across maxillary sinuses and nasal blockage for nearly a week.",
        "Nose is congested and face feels swollen and heavy, worse when leaning down.",
        "Constant facial tightness, discolored nasal phlegm, and headache around forehead.",
        "Sinus ache between eyes, stuffed nose, and mild low grade fever since 3 days.",
        "Facial pain on left cheek and forehead, tooth pain on upper jaw, and blocked nostrils.",
        "Headache with congested nose and thick mucus after recent weather change.",
        
        # Indian English colloquial
        "Naak is blocked, heavy pain on both cheeks and forehead when bending down.",
        "Facing too much sinus pain, heavy head and thick phlegm in nose since 4 days.",
        "Forehead is paining and nose is fully choked, pain increases on bending head.",
        "Nasal block with heavy pressure on face and eyes, voice is also muffled.",
        "Cold is not clearing, now heavy pain around cheekbones and forehead started.",
        "Cheeks and forehead feeling heavily congested with blocked nose and mild headache.",
        "Heavy cold with sinus pressure on forehead, pain while bending down to pray.",
        "Blocked nose, thick yellow mucus, and continuous dull facial pressure.",
        "Head is paining like sinus, forehead and cheeks are tender when touched.",
        "Severe heaviness around nose and eyes, nose blocked completely on both sides.",
        
        # Hinglish & Multilingual expressions
        "Naak poori band hai aur gaal aur forehead par bohot heavy pressure lag raha hai.",
        "Sar aage jhukane par forehead aur aankhon ke paas bohot tez dard hota hai.",
        "Sinus ki wajah se gaalon me dard aur naak se peela gaadha balgam aa raha hai.",
        "Naak me blockage hai aur aankhon ke beech me bohot daban mehsoos ho rahi hai.",
        "Mausam badalne se naak band ho gayi hai aur sar me aage bhari pan hai 4 din se.",
        "Jhukne par sar me blood rush aur forehead par bhari daban lagti hai.",
        "Naak choked hai, gale me piche kaph gir raha hai aur sar dard bana hua hai.",
        "Gaal par dabane se dard ho raha hai aur dono naak band hain.",
        "Jukham theek nahi hua aur ab forehead aur naak ke paas heavy dard shuru ho gaya.",
        "Dono taraf sinus me swelling aur bhari daban feel ho rahi hai teen din se.",
        "Naak se saans lene me dikkat aur forehead me aage thumping pain hai.",
        "Chehre par bhari pan aur naak band hone se aawaz dab gayi hai.",
        "Forehead aur aankh ke piche bohot zyada pressure lag raha hai 3 din se.",
        "Sardi ke baad naak band aur gaalon par daban wala dard shuru ho gaya hai.",
        "Naak me congestion aur sar me bohot bhari pan bana hua hai."
    ],

    "Acidity & Indigestion (Dyspepsia)": [
        # Short isolated queries
        "acidity",
        "heartburn",
        "burning in chest",
        "stomach burning",
        "sour burps",
        "gas problem",
        "indigestion",
        "chest burning after food",
        "pet me jalan",
        "pet me gas",
        "khatta dakar",

        # Conversational English & Detailed queries
        "Burning sensation in my upper chest and throat, especially after eating spicy food.",
        "Heartburn, sour belching, and feeling uncomfortably bloated after meals.",
        "Acid coming up into my throat, sour taste in mouth, and burning in stomach.",
        "Feeling overly full after eating only small amounts, upper abdomen bloating and burping.",
        "Pain and burning in epigastric region of stomach, worse when lying flat.",
        "Indigestion, severe gas formation, and burning feeling behind breastbone after dinner.",
        "Sour water coming in mouth, chest burning, and rumbling bloated stomach.",
        "Discomfort and heavy gas in upper abdomen with frequent acidic burps.",
        "Burning in chest after having oily food yesterday, difficulty sleeping flat.",
        "Stomach feels acidic, bloated and full, accompanied by mild nausea after meals.",
        "Acid regurgitation, burning feeling in food pipe, and fullness in abdomen.",
        "Gas trouble, sour tasting burps, and stomach burning for the past 2 days.",
        "Chest burning that improves after drinking cold milk or antacid syrup.",
        "Epigastric burning sensation, uncomfortable bloating, and continuous belching.",
        "Heavy burning sensation in chest after having restaurant food and tea.",
        
        # Indian English colloquial
        "Too much acidity and burning in chest after eating spicy non-veg food yesterday.",
        "Feeling heavy gas in stomach and sour burps continuously since last night.",
        "Stomach is burning and sour water is coming in throat after having fried snacks.",
        "Chest burning and acid reflux after outside food, feeling very bloated in tummy.",
        "Facing severe gas trouble, burping frequently, and burning sensation in food pipe.",
        "Food is not digesting properly, feeling heavy in upper stomach with burning chest.",
        "Acidity problem from 2 days, whenever I eat anything burning starts in chest.",
        "Burning sensation in chest and throat, feels like acid is rising up.",
        "Stomach bloated with gas and sour belches after wedding party food.",
        "Burning in chest and throat especially when lying down on bed after dinner.",
        
        # Hinglish & Multilingual expressions
        "Bahar ka spicy khana khane ke baad se pet aur seene me tez jalan ho rahi hai.",
        "Pet me bohot gas ban rahi hai aur khatte dakar aa rahe hain kal raat se.",
        "Seene me jalan aur gale tak acid aa raha hai, letne par aur badh jata hai.",
        "Khana khane ke baad pet bohot bhari lagta hai aur gale me jalan rehti hai.",
        "Pet me jalan aur bloating ho rahi hai, thoda khane par bhi pet tight lagta hai.",
        "Tea aur oily snacks lene ke baad seene me tez burning sensation ho rahi hai.",
        "Khatta paani gale me aa raha hai aur pet me gas aur daban mehsoos ho rahi hai.",
        "Seene ke beech me jalan hai par saans lene me koi dikkat nahi hai, acidity lag rahi hai.",
        "Pet me ajeeb si jalan aur gas trouble ho raha hai do din se.",
        "Spicy dinner ke baad se chest me jalan aur bar bar burping ho rahi hai.",
        "Khatta dakar aur pet me aag jaisi jalan feel ho rahi hai.",
        "Khana theek se digest nahi hua, pet fula hua hai aur seene me jalan hai.",
        "Pet me gas ghoom rahi hai aur upar chest me daban aur burning lagti hai.",
        "Thoda sa khane par bhi pet bhari aur gale tak jalan mehsoos ho rahi hai.",
        "Acidity ho gayi hai, pet aur gale me continuous burning sensation hai."
    ],

    "Viral Pharyngitis (Sore Throat)": [
        # Short isolated queries
        "sore throat",
        "throat pain",
        "pain when swallowing",
        "gale me dard",
        "gale me kharash",
        "scratchy throat",
        "throat irritation",
        "dry cough and sore throat",
        "swallowing hurts",

        # Conversational English & Detailed queries
        "I have a scratchy painful throat and it hurts whenever I swallow food or saliva.",
        "Sore throat since yesterday with difficulty swallowing and low grade tiredness.",
        "Burning irritation in the back of my throat, raspy voice, and mild dry cough.",
        "Throat feels raw, swollen and prickly, painful when drinking water or eating.",
        "Tickle in throat with sharp pain on swallowing, accompanied by mild runny nose.",
        "Raw sore throat, tender glands under jaw, and scratchy irritation since 2 days.",
        "Painful throat on swallowing, mild hoarseness, and feeling slightly feverish.",
        "Throat irritation, dry hacking cough, and discomfort in pharynx when talking.",
        "Swallowing hurts a lot, throat feels very dry, red and scratchy.",
        "Woke up with painful throat, hoarse voice, and mild body malaise.",
        "Prickly scratchy sensation in throat, worsens with cold water or air conditioning.",
        "Severe throat discomfort, tender neck, and mild congestion after cold weather.",
        "Throat feels inflamed and burning, painful swallowing, no white patches seen.",
        "Sore throat since 2 days, soothing with warm water gargles but hurts on swallowing.",
        "Irritated pharynx, scratchiness in voice box, and painful swallowing of solid food.",
        
        # Indian English colloquial
        "Throat is paining too much while swallowing water or food since yesterday.",
        "Scratchy feeling in throat and voice is becoming hoarse after drinking cold juice.",
        "Gale me pain and burning sensation while swallowing, mild body ache also there.",
        "Having bad sore throat from 2 days, warm salt water gargle gives slight relief.",
        "Throat is very rough and dry, paining while eating food or swallowing saliva.",
        "Severe irritation in throat and hoarseness of voice, feels swollen inside.",
        "Throat pain started after travel, painful to swallow and mild cough present.",
        "Feeling pricking pain in throat whenever trying to eat anything solid.",
        "Throat is feeling raw and painful since morning after sleeping in AC.",
        "Pain in throat while swallowing, dry irritation and feeling low energy.",
        
        # Hinglish & Multilingual expressions
        "Gale me bohot kharash hai aur kuch bhi nigalne me tez dard ho raha hai.",
        "Pani peene me aur khana khane me gale me chubhan aur dard lag raha hai kal se.",
        "Gala kharab hai, aawaz baith gayi hai aur nigalne me takleef ho rahi hai.",
        "Thanda paani peene ke baad se gale me tez dard aur kharash shuru ho gayi.",
        "Gale me jalan aur sujan jaisa lag raha hai, swallow karne me bohot dard hai.",
        "Do din se gala dukh raha hai, namak paani ke garare se thoda aaram milta hai.",
        "Gale ke andar kharash aur halki sukhi khansi ho rahi hai.",
        "Khana nigalte waqt gale me chubhan aur dono taraf dard hota hai.",
        "AC me sone ke baad subah se gala pack aur dukh raha hai.",
        "Gale me kanta jaisa chubhta hai jab bhi thook ya paani nigalta hoon.",
        "Gala sukha sukha lag raha hai aur bolne par aawaz phat rahi hai aur dard hai.",
        "Halka bukhar aur gale me bohot zyada kharash aur painful swallowing hai.",
        "Gale me raw feel ho raha hai aur nigalne par dono side dukh raha hai.",
        "Gala baith gaya hai aur nigalne me takleef hai do din se.",
        "Gale me redness aur continuous irritation hai, nigalna mushkil ho raha hai."
    ],

    "Dehydration & Heat Fatigue": [
        # Short isolated queries
        "dehydration",
        "extreme thirst and dizziness",
        "heat exhaustion",
        "dhoop me chakkar",
        "dry mouth and weakness",
        "dark urine and dizziness",
        "faint from heat",
        "sun fatigue",

        # Conversational English & Detailed queries
        "Feeling dizzy and lightheaded when standing up, very dry mouth and dark urine.",
        "Extreme fatigue, intense thirst, parched lips, and slight headache after being out in the sun.",
        "Dehydrated, lightheaded, feeling faint upon standing, and mouth feels completely dry.",
        "Severe weakness, dry sticky mouth, no urination for 6 hours, and muscle cramps.",
        "Sun fatigue, headache behind eyes, dizziness, and intense thirst after outdoor travel.",
        "Feeling completely drained, dry tongue, rapid pulse, and dark yellow concentrated urine.",
        "Loo exhaustion, lightheadedness, hot skin, headache, and severe lack of energy.",
        "Dry throat, parched mouth, dizziness upon standing quickly, and generalized weakness.",
        "Excessive sweating followed by dizziness, extreme thirst, and muscle fatigue.",
        "Head feels light and spinning slightly when standing, mouth is dry as paper.",
        "Exhausted after long commute in the heat, severe dehydration and low urine output.",
        "Weakness, dry skin, feeling woozy and dehydrated after playing in hot sun.",
        "Dizzy spells, dry cracked lips, constant desire to drink cold water, and fatigue.",
        "Lightheaded and sluggish, haven't drunk enough water today, mild headache.",
        "Muscle cramps in legs, extreme thirst, dry tongue, and dizziness in warm weather.",
        
        # Indian English colloquial
        "Feeling too much weakness and giddiness after going out in afternoon sun.",
        "Mouth is very dry, urine is dark yellow, and feeling dizzy whenever standing.",
        "Heat exhaustion after bike riding in sun, head is paining and feeling faint.",
        "Not drinking enough water, now feeling very dehydrated, weak and lightheaded.",
        "Body is completely drained out, dry throat, dizziness and heavy fatigue.",
        "Feeling giddiness and dry mouth after traveling in hot weather for hours.",
        "Urine color is very dark and feeling tired, thirsty and dizzy since afternoon.",
        "Too much thirst and weakness, legs are cramping after outdoor work in sun.",
        "Head spinning slightly on getting up from chair, dry lips and dehydration.",
        "Feeling hot, tired, thirsty and woozy after staying in high temperature.",
        
        # Hinglish & Multilingual expressions
        "Dhoop me ghumne ke baad se bohot chakkar aur weakness aa rahi hai, gala sukha hai.",
        "Pani kam peene ki wajah se urine dark aa raha hai aur khade hone par chakkar aate hain.",
        "Garmi aur dhoop ki wajah se sar ghoom raha hai, pyas bohot zyada lag rahi hai.",
        "Gala bilkul sookh gaya hai, hoth phat rahe hain aur sharir me bilkul jaan nahi hai.",
        "Loo lagne jaisa feel ho raha hai, weakness, dehydration aur halka sar dard hai.",
        "Khade hone par aage andhera chha jata hai aur muh poora dry hai.",
        "Bahar dhoop me bike chalane ke baad se bohot zyada thakan aur chakkar aa rahe hain.",
        "Paani ki kami ho gayi hai, sar bhari hai aur pairo me cramps aa rahe hain.",
        "Sharir me paani khatam lag raha hai, urine bohot kam aur peela aa raha hai.",
        "Bohot zyada pasina aane ke baad ab weakness, pyas aur lightheaded feel ho raha hai.",
        "Subah se dhoop me tha, ab chakkar, thakan aur gale me sukha pan hai.",
        "Muh sookh raha hai aur sar me giddiness mehsoos ho rahi hai.",
        "Garmi ki wajah se sharir toot raha hai aur bar bar pyas lag rahi hai.",
        "Chakkar aa rahe hain khade hone par, lips dry hain aur weakness hai.",
        "Dehydration ho gaya lagta hai, gale me tarawat nahi hai aur sar dukh raha hai."
    ],

    "Mechanical Low Back Strain": [
        # Short isolated queries
        "lower back pain",
        "back pain",
        "back catch",
        "stiff back",
        "kamar dard",
        "kamar me dard",
        "back pain after lifting",
        "back stiffness",

        # Conversational English & Detailed queries
        "Dull aching pain in my lower back after lifting a heavy bucket yesterday.",
        "Lower back is stiff and sore, hurts when bending forward or standing up from chair.",
        "Muscular catch in my lower spine after long hours of sitting in office chair.",
        "Ache across lumbar region of back, worse with movement, no numbness in legs.",
        "Stiff lower back, difficulty straightening up after bending down to pick up luggage.",
        "Lower back muscle spasm after motorcycle ride on bumpy roads.",
        "Dull throbbing ache in lower back after gym workout, tight glute and back muscles.",
        "Back pain localized to lower spine, relief when lying flat with knees bent.",
        "Lumbar stiffness, painful to turn or twist body, muscle tenderness in lower back.",
        "Lower back feels locked and painful after sitting on couch for several hours.",
        "Muscular tightness and soreness in lumbar back area, no tingling or loss of bladder.",
        "Lower back hurts upon waking up, takes 20 minutes of walking to loosen up.",
        "Sharp muscle catch in lower back when twisting, now dull ache remains.",
        "Aching lower spine, tired back muscles after prolonged standing and driving.",
        "Lower back fatigue and localized soreness after shifting heavy furniture.",
        
        # Indian English colloquial
        "Kamar paining too much after lifting heavy water can yesterday morning.",
        "Lower back is feeling stiff and jammed, painful to bend down or get up from sofa.",
        "Backache from continuous two-wheeler riding on bad potholed roads in Bangalore.",
        "Lower spine is aching after continuous sitting on non-ergonomic office chair.",
        "Catch in lower back while gym workout, painful to stand straight.",
        "Facing bad back pain in lower part, feeling stiffness when walking or turning.",
        "Lower back muscles are completely tight and sore after shifting house boxes.",
        "Kamar me stiffness hai, bending forward is difficult, pain stays in back only.",
        "Lower back is aching since 2 days, worse in morning and after prolonged driving.",
        "Muscular backache in lower lumbar area, resting on flat bed gives relief.",
        
        # Hinglish & Multilingual expressions
        "Kamar me bohot tez dard aur stiffness hai jhukne aur khade hone par.",
        "Bhari saman uthane ke baad se lower back me catch aa gaya hai aur dard hai.",
        "Office me lambe time tak baithne ki wajah se kamar me jhadan aur dard ho gaya hai.",
        "Bike chalane ke baad se lower back bohot dukh rahi hai, seedhe khade hone me dikkat hai.",
        "Kamar ke nichle hisse me daban aur muscle spasm hai, pair me koi dard nahi hai.",
        "Jhuk kar kaam karne ke baad se kamar me lachak aa gayi hai aur stiff lag rahi hai.",
        "Subah uthne par kamar bilkul jam lagti hai aur hilne me dard hota hai.",
        "Kamar me continuous dull dard hai 2 din se, letne par thoda aaram lagta hai.",
        "Heavy weight utha liya tha, tab se kamar me muscular dard bana hua hai.",
        "Kamar me akdan hai, niche jhuk kar chappal pehnne me bhi dard ho raha hai.",
        "Scooter chalane ke jhatko se lower back me pain start ho gaya hai.",
        "Kamar ki muscle tight ho gayi hai aur karwat lene me dard hota hai.",
        "Kamar me dard hai par pair me koi sunn pan ya tingling nahi hai.",
        "Lambe safar ke baad kamar ke nichle hisse me dukhne jaisa dard bana hua hai.",
        "Kamar seedhi karne me takleef ho rahi hai aur piche muscles me spasm hai."
    ],

    "Chikungunya": [
        # Short isolated queries
        "chikungunya",
        "joint pain after fever",
        "severe joint pain in hands",
        "fever ke baad jodo me dard",
        "post fever joint pain",
        "swollen finger joints after fever",
        "severe wrist pain post viral",

        # Conversational English & Detailed queries
        "High fever resolved 2 days ago, but now left with excruciating symmetrical joint pain in wrists and ankles.",
        "Severe joint pain and morning stiffness in finger joints, hands, and knees after high fever.",
        "Debilitating polyarthralgia in all small joints, cannot bend fingers or walk comfortably.",
        "Sudden high fever with rash, followed by persistent severe pain in wrists, knuckles, and feet.",
        "Severe symmetrical joint swelling and crippling pain in wrists and ankles following viral fever.",
        "Morning joint stiffness lasting hours, swollen finger joints, and exhaustion after mosquito bite fever.",
        "Joints in both hands and feet are aching severely and swollen after sudden fever subsided.",
        "Inability to hold objects or make a fist due to severe wrist and finger joint inflammation.",
        "Intense joint pain in ankles and knees making walking agonizing, post febrile illness.",
        "High fever with chills and now disabling multiple joint pain and faint rash on trunk.",
        "Crippling pain in hands and feet joints, stiffness worse in the morning, fever was 3 days ago.",
        "Symmetrical arthritis-like joint aches in wrists, knees, and fingers after acute high fever.",
        "Post-viral joint agony, swollen ankles, and severe difficulty getting out of bed.",
        "Extreme joint tenderness in small joints of hands and feet, can barely bend wrists.",
        "Mosquito fever followed by lingering severe joint stiffness and unbearable joint pain.",
        
        # Indian English colloquial
        "Had high fever for 3 days, now fever gone but both hands and ankle joints paining severely.",
        "Severe joint pain in fingers and wrists after fever, cannot even hold a cup of tea.",
        "Facing terrible joint stiffness in morning, feet and hands are swollen after viral fever.",
        "Joints in both legs and hands are jammed and aching continuously post fever.",
        "Cannot bend fingers and walking is very painful due to ankle joint pain after fever.",
        "Fever went down yesterday but severe crippling joint pain in all body joints started.",
        "Mosquito bite fever leading to unbearable joint pain in wrists, knees, and ankles.",
        "Both wrists and knuckles are swollen and painful to move after recent fever episode.",
        "Severe joint pain and morning stiffness, suspecting chikungunya after recent fever.",
        "Joints are paining like anything after fever, finger joints are stiff and tender.",
        
        # Hinglish & Multilingual expressions
        "Fever utarne ke baad se haath aur pairo ke joints me bohot severe dard hai.",
        "Bukhar theek ho gaya par dono kalaiyon aur ungliyon ke jodo me aisi akdan hai ki mutthi nahi band ho rahi.",
        "Ghutno, takhno aur haath ke jodo me bohot tez dard hai, chalne me bohot takleef ho rahi hai.",
        "Subah uthne par joints bilkul jam rehte hain aur ghanto tak dard rehta hai fever ke baad.",
        "Macchar katne ke baad tez bukhar aaya tha, ab sharir ke saare jodo me ajeeb sa dard hai.",
        "Chikungunya jaisa lag raha hai, bukhar ke baad se fingers aur wrists me severe swelling aur pain hai.",
        "Haath pair ke saare joints dukh rahe hain, chammach pakadne me bhi dard ho raha hai.",
        "Bukhar do din pehle utra par jodo ka dard badhta hi ja raha hai dono taraf.",
        "Ungliyon ke joint soojh gaye hain aur morning me bilkul akad jate hain.",
        "Bohot tez bukhar ke baad ab saare joints me crippling dard aur halki red rash hai.",
        "Pairo ke takhno aur ghutno me itna dard hai ki bina sahare khada nahi hua ja raha.",
        "Joints me severe inflammation aur dard hai, bukhar ke baad haath pair kaam nahi kar rahe.",
        "Dono hatho ki kalai me bohot tez dukhawa hai fever ke baad se.",
        "Fever khatam hone ke baad jodo me severe swelling aur chalne me lachak aa rahi hai.",
        "Saare chhote-bade joints me unbearable dard bana hua hai viral ke baad."
    ]
}

def generate_synthetic_dataset():
    records = []
    for disease, samples in SYNTHETIC_CONDITIONS_DATA.items():
        for text in samples:
            records.append({
                "disease": disease,
                "symptom_text": text.strip()
            })
            
    df = pd.DataFrame(records)
    print(f"Generated {len(df)} synthetic samples across {df['disease'].nunique()} conditions:")
    for d, count in df["disease"].value_counts().items():
        print(f"  - {d}: {count} samples")
        
    out_path = OUTPUT_DIR / "outpatient_indian_symptoms.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved synthetic dataset to: {out_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_dataset()
