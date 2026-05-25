# Twitter Sentiment Analysis con LSTM (Matlab)

Questo progetto implementa una rete neurale ricorrente di tipo **Long Short-Term Memory (LSTM)** in ambiente **MATLAB** per l'analisi del sentiment di file testuali contenenti dei possibili commenti presi da Twitter. L'obiettivo principale è la **costruzione da zero dell'architettura LSTM**, senza l'ausilio di toolbox di machine learning o deep learning predefiniti.

📄 **[Relazione del progetto](./Relazione_Metodi_di_ottimizzazione_Big_Data.pdf)**

## Scopo del Progetto
Il progetto si focalizza sulla classificazione binaria del sentiment (positivo/negativo) utilizzando un dataset di tweet pre-elaborati. Di particolare importanza vi è l'implementazione manuale di `forward` e `backward` pass e della gestione degli stati della cella LSTM. Conseguentemente ad essi, vengono esplorati e confrontati diversi algoritmi di ottimizzazione per l'addestramento della rete sul dataset reale (Twitter), validando l'efficacia della struttura implementata.

## Architettura del Sistema
Il workflow è suddiviso in fasi logiche:

1.  **Preprocessing**: Pulizia dei tweet, tokenizzazione e creazione di un vocabolario di 10000 parole.
2.  **Inizializzazione**: Utilizzo di pesi **Xavier** standardizzati per garantire che tutti i modelli partano dalla stessa base, permettendo un confronto equo.
3.  **Architettura LSTM**: Implementazione manuale, senza toolbox di machine learning, di `forward` e `backward` pass.
4.  **K-Fold Cross Validation**: Implementazione di un meccanismo di Cross Validation per individuare i migliori iperparametri di training.
5.  **Algoritmi di ottimizzazione**: Implementazione di quattro varianti principali per testare la robustezza dell'architettura costruita:
    *   **SGD Nesterov**: Con learning rate dinamico.
    *   **Adagrad**: Scaling adattivo per gestire la sparsità dei dati.
    *   **RMSprop**: Media mobile per stabilizzare l'apprendimento.
    *   **AdamW**: Con correzione del bias e Weight Decay.

## Struttura della Repository
*   `load_twitterdata.m`: Script per caricare e processare il dataset.
*   `analyze_seq_length.m`: Analisi statistica della lunghezza dei tweet per ottimizzare il padding.
*   `init_lstm.m`: Inizializzazione dei parametri della rete.
*   `forward_lstm.m` / `backward_lstm.m`: Logica della rete LSTM.
*   `train_lstm_*.m`: Script di addestramento della rete per ogni algorimto di ottimizzazione.
*   `train_fold_generic.m`: Motore di addestramento utilizzato per la Cross Validation.
*   `kfold_gridsearch.m`: Script per la selezione dei migliori iperparametri per ogni algoritmo di ottimizzazione.
*   `confronto_finale.m`: Script per la generazione di grafici comparativi (Loss e Accuracy) tra i diversi ottimizzatori.
*   `test_lstm.m`: Valutazione del modello (Matrice di Confusione, Precision, Recall, F1-Score) sul test set.
*   `test_custom_tweets.m`: Script per testare il modello su frasi arbitrarie (con parole appartenenti al vocabolario).

## Requisiti
*   MATLAB (R2021a o superiore consigliato).
*   **Parallel Computing Toolbox** (necessario per la Grid Search accelerata).

## Guida all'Uso

### 1. Preparazione Dati
1.  Scaricare la cartella estratta del dataset grezzo `TwitterParsed` da **[questo link su Google Drive](https://drive.google.com/file/d/1NtPI59J3pAmvZfQzjZ2oz9UGIPxzj9Yd/view?usp=sharing)** e posizionarla nella directory principale del progetto.
2.  (Opzionale) Eseguire `analyze_seq_length.m` per visualizzare la distribuzione delle lunghezze dei tweet.
3.  Eseguire lo script `load_twitterdata.m` che applicherà la pipeline di pulizia e preprocessing del dataset, generando localmente il file compilato `ready_data.mat` utilizzato dagli script successivi.

### 2. Inizializzazione
Eseguire `init_lstm.m` per inizializzare la rete la matrice di pesi iniziali standardizzati, che verrà salvata localmente nel file `init_lstm.mat`.

### 3. Ricerca Iperparametri
Per trovare la combinazione migliore di Learning Rate e Lambda si deve eseguire lo script `kfold_gridsearch.m`. Di default il parametro `OPTIMIZER` è impostato su `'sgd'`; per testare gli altri algoritmi, è sufficiente modificare la stringa in cima allo script selezionando una delle opzioni supportate: `'sgd'`, `'adagrad'`, `'rmsprop'`, o `'adamw'`. Lo script testerà diverse combinazioni tramite Cross-Validation a 3 fold, delegando l'addestramento dei singoli fold alla funzione `train_fold_generic.m`.

### 4. Training Finale
Aggiornare i parametri negli script `train_lstm_*.m` con i valori ottimali trovati (o usare quelli predefiniti) e avviare l'addestramento. Al termine di ogni epoca lo script valuterà la rete sul validation set, salvando su disco il checkpoint con il modello migliore.

### 5. Valutazione e Test
*   **Confronto Modelli**: Eseguire `confronto_finale.m` per visualizzare l'andamento della Loss e dell'Accuracy di validazione di tutti gli ottimizzatori addestrati.
*   **Performance Dettagliate**: Eseguire `test_lstm.m` per analizzare il modello scelto sul Test Set. Lo script produrrà la Matrice di Confusione e calcolerà Precision, Recall e F1-Score.
*   **Test Interattivo**: Utilizzare `test_custom_tweets.m` per inserire frasi a piacere e verificare la classificazione del sentiment su frasi mai viste (l'efficienza dipende fortemente da parole presenti nel vocabolario creato).
