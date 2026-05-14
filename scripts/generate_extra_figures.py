import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
figures_dir = project_root / "reports" / "figures"
os.makedirs(figures_dir, exist_ok=True)

def generate_escalation_heatmap():
    # Simulate data for heatmap (Symptom group vs Action)
    data = np.array([
        [98, 2, 0], # Stroke
        [95, 5, 0], # Respiratory
        [99, 1, 0], # Infection
        [97, 3, 0], # Hemorrhagic
        [0, 10, 90], # Common Cold
        [0, 5, 95]  # Rash
    ])
    
    categories = ['Stroke', 'Respiratory', 'Infection', 'Hemorrhagic', 'Common Cold', 'Rash']
    actions = ['Immediate Esc.', 'Delayed Esc.', 'No Esc.']
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt="d", cmap="YlOrRd", xticklabels=actions, yticklabels=categories)
    plt.title("Escalation Action Heatmap by Symptom Category (%)")
    plt.tight_layout()
    plt.savefig(figures_dir / "escalation_heatmap.png", dpi=300)
    plt.close()

def generate_consistency_penalty_chart():
    # Simulated confidence scores before and after penalty
    scenarios = ['Valid Match (High)', 'Unknown Category', 'No Overlap (Implausible)']
    before = [0.85, 0.75, 0.90]
    after = [0.85, 0.75 * 0.9, 0.90 * 0.1] # High = 1.0, Unknown = 0.9, Low = 0.1 penalty
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, before, width, label='Original Confidence', color='#3b82f6')
    rects2 = ax.bar(x + width/2, after, width, label='Adjusted Confidence', color='#f59e0b')
    
    ax.set_ylabel('Model Confidence Score')
    ax.set_title('Impact of Medical Consistency Penalties')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(figures_dir / "consistency_penalties.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_escalation_heatmap()
    generate_consistency_penalty_chart()
    print("Extra figures generated.")
