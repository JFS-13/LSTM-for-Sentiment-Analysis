# Stato del Progetto: Sentiment Analysis Twitter (Matlab)

## Architettura e Workflow Consolidato
1.  **Preprocessing:** `load_twitterdata.m` -> Genera `ready_data.mat` (Vocab: 5000 parole, Coverage: 86.79%).
2.  **Inizializzazione:** `init_lstm.m` -> Crea `init_lstm.mat`. **Tutti i modelli partono ora dallo stesso set di pesi Xavier** per garantire confronti scientificamente validi.
3.  **Ottimizzazione (Standardizzata):**
    *   **SGD Nesterov**: `train_lstm_sgd_momentum.m` (Robbins-Monro $\alpha_k$, Momentum 0.95).
    *   **Adagrad**: `train_lstm_adagrad.m` (Scaling diagonale adattivo).
    *   **RMSprop**: `train_lstm_rmsprop.m` (Media mobile dei quadrati).
    *   **AdamW**: `train_lstm_adamW.m` (Bias correction e Decoupled Weight Decay).
4.  **Validazione:** `kfold_gridsearch.m` e `train_fold_generic.m` per la ricerca sistematica di $\lambda$.

## Risultati Finali (Benchmarking)
Valutazione effettuata tramite `confronto_finale.m` e script di test individuali (`test_lstm_best_*.m`):

| Metodo | Accuracy (Test) | F1-Score | Osservazioni |
| :--- | :--- | :--- | :--- |
| **SGD Nesterov** | 77.15% | 0.7620 | Stabile, richiede più epoche. |
| **Adagrad** | 77.72% | 0.7782 | Convergenza rapidissima, tendenza all'overfitting precoce. |
| **RMSprop** | 78.05% | 0.7864 | **Miglior compromesso stabilità/accuratezza.** |
| **AdamW** | 76.79% | 0.7715 | Molto sensibile agli iperparametri, ottima loss iniziale. |

## Strumenti di Analisi Qualitativa
*   **`test_custom_tweets.m`**: Script per il test interattivo di frasi arbitrarie (simulazione casi reali).
*   **`confronto_finale.m`**: Generazione automatica di grafici comparativi (Loss e Validation Accuracy) con marker distinti per analisi sovrapposizioni.

## Finalizzazione Style e Ottimizzazione Grid Search
*   **Refactoring Accademico:** Tutti i file `.m` sono stati aggiornati con header ufficiali e documentazione dei parametri conforme allo stile richiesto.
*   **Grid Search Potenziata (LR + Lambda):** Lo script `kfold_gridsearch.m` ora ottimizza simultaneamente sia il **Learning Rate** che il **Weight Decay**.
*   **Parallelizzazione (`parfor`):** Implementata la ricerca parallela per sfruttare i core della CPU e abbattere i tempi di calcolo.
*   **Strategie di Efficienza:** Per la fase di ricerca iperparametri, sono stati impostati `K_FOLDS = 3` e `EPOCHS = 3` in `train_fold_generic.m`.

### Prossimi Passi (Handoff per ripresa sessione)
1.  **Verifica Toolbox:** Assicurarsi che il *Parallel Computing Toolbox* sia installato (`ver` in Command Window).
2.  **Lancio Grid Search:** Eseguire `kfold_gridsearch.m`. Lo script avvierà automaticamente il `parpool`.
3.  **Analisi Risultati:** Al termine, analizzare la superficie 3D generata e i valori ottimali stampati in console.
4.  **Training Finale:** Aggiornare gli iperparametri negli script di training specifici (es. `train_lstm_adamW_optimized.m`) con i valori "vincitori" trovati e lanciare il training completo (10 epoche).

---
*Ultimo aggiornamento: 7 Maggio 2026 (Sistema pronto per Grid Search Parallela)*
