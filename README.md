# Twitter Sentiment Analysis con LSTM (Matlab)

Questo progetto implementa una rete neurale ricorrente di tipo **Long Short-Term Memory (LSTM)** in ambiente **MATLAB** per l'analisi del sentiment di file testuali contenenti dei possibili commenti presi da Twitter. L'obiettivo principale è la **costruzione da zero dell'architettura LSTM**, senza l'ausilio di toolbox di machine learning o deep learning predefiniti.

## Scopo del Progetto
Il progetto si focalizza sulla classificazione binaria del sentiment (positivo/negativo) utilizzando un dataset di tweet pre-elaborati. Di particolare importanza vi è l'implementazione manuale dei passaggi di `forward` e `backward` propagation e della gestione degli stati della cella LSTM. Conseguentemente ad essi, vengono esplorati e confrontati diversi algoritmi di ottimizzazione per l'addestramento della rete sul dataset reale (Twitter), validando l'efficacia della struttura implementata.

## Architettura del Sistema
Il workflow è suddiviso in fasi logiche:

1.  **Preprocessing**: Pulizia dei tweet, tokenizzazione e creazione di un vocabolario di 5000 parole (copertura ~86.8%).
2.  **Inizializzazione**: Utilizzo di pesi **Xavier** standardizzati per garantire che tutti i modelli partano dalla stessa base, permettendo un confronto equo.
3.  **Core LSTM**: Implementazione manuale, senza toolbox di machine learning, dei passaggi di `forward` e `backward` propagation.
4.  **Algoritmi di ottimizzazione**: Implementazione di quattro varianti principali per testare la robustezza dell'architettura costruita:
    *   **SGD Nesterov**: Con learning rate dinamico.
    *   **Adagrad**: Scaling adattivo per gestire la sparsità dei dati.
    *   **RMSprop**: Media mobile per stabilizzare l'apprendimento.
    *   **AdamW**: Con correzione del bias e Weight Decay.

## Struttura della Repository
*   `load_twitterdata.m`: Script per caricare e processare il dataset.
*   `init_lstm.m`: Inizializzazione dei parametri della rete.
*   `forward_lstm.m` / `backward_lstm_optimized.m`: Logica core della rete LSTM.
*   `train_lstm_*.m`: Script specifici per ogni metodo di ottimizzazione.
*   `kfold_gridsearch_generic.m`: Script per la selezione dei migliori iperparametri per ogni algoritmo di ottimizzazione.
*   `confronto_finale.m`: Visualizzazione e confronto tra i risultati ottenuti con i diversi ottimizzatori.
*   `test_custom_tweets.m`: Script per testare il modello su frasi arbitrarie.

## Requisiti
*   MATLAB (R2021a o superiore consigliato).
*   **Parallel Computing Toolbox** (necessario per la Grid Search accelerata).

## Guida all'Uso

### 1. Preparazione Dati
Eseguire `load_twitterdata.m` per generare il file `ready_data.mat`. Assicurarsi che la cartella `TwitterParsed` o il relativo `.zip` siano presenti.

### 2. Inizializzazione
Eseguire `init_lstm.m` per generare i pesi iniziali `init_lstm.mat`.

### 3. Ricerca Iperparametri
Per trovare la combinazione migliore di Learning Rate e Lambda:
```matlab
kfold_gridsearch_generic
```
Lo script utilizzerà i core della CPU per testare diverse combinazioni tramite Cross-Validation a 3 fold.

### 4. Training Finale
Aggiornare i parametri negli script `train_lstm_*.m` con i valori ottimali trovati e avviare l'addestramento (10 epoche impostate).

### 5. Valutazione e Test
*   Eseguire `confronto_finale.m` per generare i grafici di Loss e Accuracy.
*   Utilizzare `test_custom_tweets.m` per una verifica su input manuali.
