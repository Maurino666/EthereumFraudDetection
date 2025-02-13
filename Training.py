import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

#Carica il dataset
df = pd.read_csv('data/Prepared_Data.csv')

#Divisione tra variabili indipendenti e dipendenti
x = df.drop(columns=['FLAG', 'ERC20_most_sent_token_type_encoded', 'ERC20_most_rec_token_type_encoded'])
y = df['FLAG']

#Divisione tra training e test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

corr_matrix = df.corr()
print(corr_matrix["FLAG"].sort_values(ascending=False))


# Calcolo dei pesi delle classi
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Addestramento del modello con class_weight
model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight={0: 1, 1: 2})

model.fit(X_train, y_train)

# Predizioni e valutazione
y_pred = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilità di essere True
y_pred_adj = (y_pred_proba > 0.3).astype(int)  # Cambiamo la soglia
print("Classification Report:\n", classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred_adj))


