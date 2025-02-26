from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

#Carica il dataset
df = pd.read_csv('data/data_preparation_results/Prepared_Data.csv')

#Divisione tra variabili indipendenti e dipendenti
y = df['FLAG']
x = df.drop(columns=['FLAG', 'ERC20_most_sent_token_type_encoded', 'ERC20_most_rec_token_type_encoded'])

print(x.columns)

#Divisione tra training e test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#Normalizzazione delle variabili indipendenti
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
# Utilizziamo la stessa normalizzazione per il test set
x_test_scaled = scaler.transform(X_test)

# Riconvertiamo in DataFrame per mantenere i nomi delle feature
x_train_scaled = pd.DataFrame(x_train_scaled, columns=X_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=X_test.columns)

# Calcolo dei pesi delle classi
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Addestramento del modello con class_weight
model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight={0: 1, 1: 2})

model.fit(x_train_scaled, y_train)

# Predizioni e valutazione
y_pred = model.predict(x_test_scaled)

# Generiamo il nome del file con data e ora
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"results/classification_results_{current_time}.txt"

y_pred_proba = model.predict_proba(x_test_scaled)[:, 1]  # Probabilità di essere True

train_accuracy = model.score(x_train_scaled, y_train)
test_accuracy = model.score(x_test_scaled, y_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Scriviamo i risultati su file
with open(results_file, "w") as f:
    f.write("Training Accuracy: {:.4f}\n".format(train_accuracy))
    f.write("Risultati della classificazione con rimozione di feature altamente correlate e senza feature aggiuntive:\n\n")
    for i in range(1, 6):
        threshold = 0.1 * i
        y_pred_adj = (y_pred_proba > threshold).astype(int)  # Cambiamo la soglia
        report = classification_report(y_test, y_pred_adj)
        f.write(f"Classification Report [soglia = {threshold}]:\n{report}\n\n")
    f.write("Analisi completata con successo.\n")


print(f"I risultati sono stati salvati in {results_file}")


