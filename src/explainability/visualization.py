import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.classical_ml.config import config

def plot_local_explanation(explanation_dict, model_name="model"):
    """Plot explanation from explain_prediction.py."""
    if "explanations" not in explanation_dict:
        return
        
    exps = explanation_dict["explanations"]
    if not exps:
        return
        
    words = [e['word'] for e in exps[:10]]
    contribs = [e['contribution'] for e in exps[:10]]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=contribs, y=words, palette="Blues_d")
    plt.title(f"Local Explanation for: '{explanation_dict['text'][:30]}...'")
    plt.xlabel("Contribution Score")
    plt.ylabel("Word")
    
    figures_dir = os.path.join(config.PROJECT_ROOT, "reports/figures/explainability")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Use a safe filename
    safe_text = "".join([c if c.isalnum() else "_" for c in explanation_dict['text'][:10]])
    plt.savefig(os.path.join(figures_dir, f"{model_name}_local_exp_{safe_text}.png"), bbox_inches='tight')
    plt.close()
