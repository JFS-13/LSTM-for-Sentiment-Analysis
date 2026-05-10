% Test LSTM

% This script evaluates the performance of a selected LSTM model on the 
% full test dataset. It calculates key metrics such as Accuracy, F1-Score, 
% Precision, and Recall, and displays a confusion matrix to analyze 
% classification results.

clear; clc;

% --- 1. CONFIGURATION ---
% select the model to test: 'adagrad', 'rmsprop', 'adamw', 'sgd'
MODELLO = 'adamw'; 

switch lower(MODELLO)
    case 'adagrad'
        file_mat = 'best_lstm_adagrad.mat';
        label = 'ADAGRAD';
    case 'rmsprop'
        file_mat = 'best_lstm_rmsprop.mat';
        label = 'RMSPROP';
    case 'adamw'
        file_mat = 'best_lstm_adamW_optimized.mat';
        label = 'ADAMW';
    case 'sgd'
        file_mat = 'best_lstm_SGD_momentum.mat';
        label = 'SGD';
    otherwise
        error('Modello non riconosciuto.');
end

% --- 2. DATA LOADING ---
fprintf('Caricamento dati Test e Rete LSTM (%s)...\n', label);
if isfile('ready_data.mat') && isfile(file_mat)
    load('ready_data.mat', 'X_Test', 'Y_Test');
    load(file_mat); 
else
    error('File mancanti! Controlla di aver eseguito il training.');
end

num_test = size(X_Test, 1);
BATCH_SIZE = 64; 

% metrics initialization
correct = 0; TP = 0; TN = 0; FP = 0; FN = 0;
t_start = tic;

% --- 3. EVALUATION LOOP ---
for i = 1:BATCH_SIZE:num_test
    idx_end = min(i + BATCH_SIZE - 1, num_test);
    x_batch = X_Test(i:idx_end, :);
    y_batch = Y_Test(i:idx_end);
    
    [prob, ~] = forward_lstm(x_batch, lstm_net, 0);
    pred = round(prob);
    
    for k = 1:length(y_batch)
        if pred(k) == y_batch(k)
            correct = correct + 1;
            if y_batch(k) == 1, TP = TP + 1; else, TN = TN + 1; end
        else
            if y_batch(k) == 1, FN = FN + 1; else, FP = FP + 1; end
        end
    end
end

% --- 4. METRICS CALCULATION ---
acc = (correct / num_test) * 100;
prec = TP / (TP + FP + 1e-10);
rec = TP / (TP + FN + 1e-10);
f1 = 2 * (prec * rec) / (prec + rec + 1e-10);

% --- 5. RESULTS OUTPUT ---
fprintf('\n=====================================\n');
fprintf('   RISULTATI TEST: %s\n', label);
fprintf('=====================================\n');
fprintf('Accuratezza: %.2f%%\n', acc);
fprintf('F1-Score:    %.4f\n', f1);
fprintf('Precision:   %.4f\n', prec);
fprintf('Recall:      %.4f\n', rec);
fprintf('Tempo:       %.2f s\n', toc(t_start));
fprintf('-------------------------------------\n');
fprintf('Matrice di Confusione:\n');
fprintf('               Pred 0      Pred 1\n');
fprintf('Reale 0        %-10d  %-10d\n', TN, FP);
fprintf('Reale 1        %-10d  %-10d\n', FN, TP);
fprintf('=====================================\n');
