import json
import os

def create_notebook(filename, title, description, cells_data):
    """
    cells_data: list of tuples (type, content)
    """
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Add title
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n", f"{description}"]
    })
    
    for c_type, content in cells_data:
        # Convert content to list of strings ending with \n
        lines = content.splitlines(keepends=True)
        if c_type == "code":
            nb["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            })
        else:
            nb["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines
            })
            
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

# --- Symptom Notebook ---
symptom_cells = [
    ("markdown", "## 1. Environment Setup\nLoading essential libraries and setting visualization parameters."),
    ("code", "import os\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom collections import Counter\n%matplotlib inline\n\n# Aesthetics\nsns.set(style='whitegrid', palette='muted')\nplt.rcParams['figure.figsize'] = (12, 8)\nplt.rcParams['font.size'] = 12"),
    ("markdown", "## 2. Data Loading and Initial Inspection"),
    ("code", "RAW_PATH = '../../data/raw/symptoms/final_symptoms_to_disease.csv'\ndf = pd.read_csv(RAW_PATH)\nprint(f'Dataset Shape: {df.shape}')\ndf.head()"),
    ("markdown", "## 3. Disease Label Analysis\nAnalyzing class distribution and detecting potential imbalance."),
    ("markdown", "### Pre-generated Distribution\n![Disease Distribution](../../reports/figures/symptoms_disease_dist.png)"),
    ("code", "disease_counts = df['diseases'].value_counts()\nprint(f'Total unique diseases: {len(disease_counts)}')\n\n# Visualize Top 30 Diseases\ndisease_counts[:30].plot(kind='bar', color='skyblue')\nplt.title('Top 30 Diseases by Frequency')\nplt.ylabel('Number of Samples')\nplt.xticks(rotation=45, ha='right')\nplt.show()"),
    ("markdown", "## 4. Symptom Extraction and Frequency"),
    ("markdown", "### Pre-generated Top Symptoms\n![Top Symptoms](../../reports/figures/symptoms_top_symptoms.png)"),
    ("code", "all_symptoms = []\ndf['symptom_list'] = df['symptom_text'].apply(lambda x: [s.strip().lower() for s in str(x).split(',')])\nfor s_list in df['symptom_list']:\n    all_symptoms.extend(s_list)\n\nsymptom_counts = Counter(all_symptoms)\nsymptom_df = pd.DataFrame(symptom_counts.most_common(30), columns=['Symptom', 'Count'])\n\nsns.barplot(data=symptom_df, x='Count', y='Symptom', palette='viridis')\nplt.title('Top 30 Most Frequent Symptoms')\nplt.show()"),
    ("markdown", "## 5. Advanced Analysis: Symptom Co-occurrence"),
    ("markdown", "### Pre-generated Co-occurrence Matrix\n![Co-occurrence](../../reports/figures/symptoms_cooccurrence.png)"),
    ("code", "top_20_symptoms = [s for s, c in symptom_counts.most_common(20)]\nco_matrix = np.zeros((20, 20))\nfor s_list in df['symptom_list']:\n    for i in range(20):\n        for j in range(20):\n            if top_20_symptoms[i] in s_list and top_20_symptoms[j] in s_list:\n                co_matrix[i, j] += 1\n\nsns.heatmap(co_matrix, xticklabels=top_20_symptoms, yticklabels=top_20_symptoms, annot=True, fmt='g', cmap='YlGnBu')\nplt.title('Symptom Co-occurrence Heatmap')\nplt.show()"),
    ("markdown", "## 6. Disease Overlap (Jaccard Similarity)"),
    ("markdown", "### Pre-generated Similarity Matrix\n![Disease Similarity](../../reports/figures/symptoms_disease_similarity.png)"),
    ("code", "disease_symptom_map = df.groupby('diseases')['symptom_list'].apply(lambda x: set().union(*x)).to_dict()\ndiseases = list(disease_symptom_map.keys())\n\n# Analyze Top 20 for visibility\nsub_diseases = diseases[:20]\noverlap_matrix = np.zeros((20, 20))\nfor i in range(20):\n    for j in range(20):\n        s1 = disease_symptom_map[sub_diseases[i]]\n        s2 = disease_symptom_map[sub_diseases[j]]\n        if len(s1.union(s2)) > 0:\n            overlap_matrix[i, j] = len(s1.intersection(s2)) / len(s1.union(s2))\n\nsns.heatmap(overlap_matrix, xticklabels=sub_diseases, yticklabels=sub_diseases, annot=True, fmt='.2f', cmap='Reds')\nplt.title('Disease Symptom Overlap (Jaccard Similarity)')\nplt.show()"),
    ("markdown", "## 7. Conclusions and Model Implications\n- **Imbalance**: Class weights or oversampling may be needed for rare diseases.\n- **Ambiguity**: High overlap suggests Top-K prediction is safer than Top-1.\n- **Sparsity**: Many symptoms are rare; TF-IDF helps focus on distinguishing features.")
]
create_notebook("notebooks/01_data_analysis/symptom_dataset_eda.ipynb", "Symptom Dataset EDA", "In-depth investigation of medical symptom mapping and disease classification feasibility.", symptom_cells)

# --- Conversation Notebook ---
conv_cells = [
    ("markdown", "## 1. Environment and Data Loading"),
    ("code", "import json\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom wordcloud import WordCloud\n%matplotlib inline\n\nsns.set(style='whitegrid')"),
    ("code", "RAW_PATH = '../../data/raw/conversations/HealthCareMagic-100k.json'\nwith open(RAW_PATH, 'r') as f:\n    data = json.load(f)\ndf = pd.DataFrame(data)\nprint(f'Total Conversations: {len(df)}')\ndf.head()"),
    ("markdown", "## 2. Text Length Distribution"),
    ("markdown", "### Pre-generated Length Distribution\n![Length Distribution](../../reports/figures/conv_length_dist.png)"),
    ("code", "df['input_len'] = df['input'].apply(lambda x: len(str(x).split()))\ndf['output_len'] = df['output'].apply(lambda x: len(str(x).split()))\n\nplt.figure(figsize=(12, 6))\nsns.histplot(df['input_len'], bins=100, color='blue', label='Patient Input', kde=True)\nsns.histplot(df['output_len'], bins=100, color='green', label='Doctor Output', kde=True)\nplt.xlim(0, 500)\nplt.title('Token Count Distribution')\nplt.legend()\nplt.show()"),
    ("markdown", "## 3. Linguistic Visualization: Word Clouds"),
    ("markdown", "### Pre-generated Word Cloud\n![Input Wordcloud](../../reports/figures/conv_wordcloud_input.png)"),
    ("code", "sample_text = ' '.join(df['input'].astype(str).sample(5000))\nwordcloud = WordCloud(width=800, height=400, background_color='white').generate(sample_text)\n\nplt.figure(figsize=(15, 8))\nplt.imshow(wordcloud, interpolation='bilinear')\nplt.axis('off')\nplt.title('Patient Query Word Cloud')\nplt.show()"),
    ("markdown", "## 4. Vocabulary Analysis"),
    ("markdown", "### Pre-generated Top Words\n![Top Words](../../reports/figures/conv_top_words.png)"),
    ("code", "from collections import Counter\nimport re\n\ndef get_tokens(text):\n    return re.findall(r'\\w+', str(text).lower())\n\nall_tokens = []\nfor t in df['input'].sample(10000): all_tokens.extend(get_tokens(t))\n\n# Filter common stopwords\nsw = {'the', 'and', 'i', 'to', 'a', 'is', 'have', 'of', 'in', 'it', 'my', 'for'}\nfiltered = [w for w in all_tokens if w not in sw]\n\ncounts = Counter(filtered)\ntop_20 = pd.DataFrame(counts.most_common(20), columns=['Word', 'Freq'])\nsns.barplot(data=top_20, x='Freq', y='Word', palette='magma')\nplt.title('Top 20 Frequent Medical Terms (Filtered)')\nplt.show()"),
    ("markdown", "## 5. NLP Model Readiness (BERT/DistilBERT)\n- **Realism**: High variety in expression supports robust NLU.\n- **Symmetry**: Balanced input/output lengths suggest good sequence-to-sequence potential.\n- **Noise**: Typos and shorthand exist, requiring robust tokenization (WordPiece).")
]
create_notebook("notebooks/01_data_analysis/conversation_dataset_eda.ipynb", "Conversation Dataset EDA", "NLP-centric analysis of patient-doctor medical interactions.", conv_cells)

# --- Hospital Notebook ---
hosp_cells = [
    ("markdown", "## 1. Data Loading\nReading the national hospital directory and local rural health datasets."),
    ("code", "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n%matplotlib inline"),
    ("code", "HOSP_PATH = '../../data/raw/hospitals/hospital_directory.csv'\nRURAL_PATH = '../../data/raw/rural_health/Health_Care_Facilities_Tiruppur_0.csv'\n\ntry:\n    df_hosp = pd.read_csv(HOSP_PATH, encoding='utf-8')\nexcept:\n    df_hosp = pd.read_csv(HOSP_PATH, encoding='cp1252')\n\ndf_rural = pd.read_csv(RURAL_PATH)\nprint('Data loaded successfully.')"),
    ("markdown", "## 2. Geographic Distribution"),
    ("markdown", "### Pre-generated State Distribution\n![State Distribution](../../reports/figures/hosp_state_dist.png)"),
    ("code", "df_hosp['State'].value_counts().plot(kind='bar', figsize=(15, 7), color='teal')\nplt.title('Hospital Density by State')\nplt.ylabel('Count')\nplt.show()"),
    ("markdown", "## 3. Hospital Categorization"),
    ("markdown", "### Pre-generated Category Distribution\n![Category Distribution](../../reports/figures/hosp_category_pie.png)"),
    ("code", "df_hosp['Hospital_Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(10, 10))\nplt.title('Facility Ownership Distribution')\nplt.ylabel('')\nplt.show()"),
    ("markdown", "## 4. Rural Facility Geo-Analysis"),
    ("markdown", "### Pre-generated Geo Distribution\n![Geo Distribution](../../reports/figures/rural_geo_dist.png)"),
    ("code", "if 'LATITUDE' in df_rural.columns:\n    plt.figure(figsize=(10, 8))\n    sns.scatterplot(data=df_rural, x='LONGITUDE', y='LATITUDE', hue='Govt and Private', alpha=0.7)\n    plt.title('Tiruppur Rural Health Facility Distribution')\n    plt.show()"),
    ("markdown", "## 5. Recommendation System Insights\n- **Coverage**: High national coverage for state-level recommendations.\n- **Geo-Precision**: Local rural data is much richer in coordinate data than the national set.\n- **Metadata**: Categories are well-defined, supporting filtered recommendations.")
]
create_notebook("notebooks/01_data_analysis/hospital_dataset_eda.ipynb", "Hospital Dataset EDA", "Healthcare infrastructure and availability analysis.", hosp_cells)
