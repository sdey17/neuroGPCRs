"""
Simple example of using the prediction script.

This example shows how to predict GPCR-ligand interactions for a set of
compounds given as SMILES strings.
"""

import sys
sys.path.insert(0, '..')

from scripts.predict_interactions import load_gpcrs_from_training, predict_interactions

# Example SMILES strings
smiles_list = [
    "CCOCc1sc(NC(=O)c2ccco2)nc1-c1ccccc1",  # Example compound 1
    "COc1cc(N(C)CCN(C)C)c2nc(C(=O)Nc3ccc(N4CCOCC4)cc3)cc(O)c2c1",  # Example compound 2
    "COc1ccccc1OCCNCCCc1c[nH]c2ccccc12"  # Example compound 3
]

# Load GPCRs from training set
gpcrs_df = load_gpcrs_from_training("../data/training_set.csv")

# Make predictions
# NOTE: You need to provide the path to your trained model
model_path = "../results/cross_attention_prot_frozen_mol_frozen_run1.pth"  # Adjust this path

results_df = predict_interactions(
    smiles_list=smiles_list,
    gpcrs_df=gpcrs_df,
    model_path=model_path,
    batch_size=16
)

# Show top predictions for each compound
print("\nTop 5 predictions for each compound:")
print("="*80)
for smiles in smiles_list:
    print(f"\nSMILES: {smiles}")
    compound_results = results_df[results_df['SMILES'] == smiles].head(5)
    print(compound_results[['UniProt', 'Binding_Probability']].to_string(index=False))
    print("-"*80)

# Save all predictions
results_df.to_csv("example_predictions.csv", index=False)
print("\nAll predictions saved to: example_predictions.csv")
