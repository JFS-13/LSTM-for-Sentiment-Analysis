% Test LSTM Best SGD

% This script evaluates the best LSTM model trained using the SGD Nesterov 
% optimization method. It calculates the final test accuracy, F1-Score, 
% and displays the confusion matrix.

clear; clc;

% --- 1. DATA LOADING ---
disp('Caricamento dati Test e Rete LSTM SGD Nesterov Ottimizzata...');
if isfile('ready_data.mat') && isfile('best_lstm_SGD_momentum.mat')
    load('ready_data.mat', 'X_Test', 'Y_Test');
    load('best_lstm_SGD_momentum.mat'); 
else
    error('File mancanti! Esegui prima load_twitterdata_optimized e lstm_train_SGD_momentum.');
end

num_test = size(X_Test, 1);
BATCH_SIZE = 32; 

% metrics initialization
correct = 0;
TP = 0; TN = 0; FP = 0; FN = 0;

fprintf('Valutazione (SGD Nesterov) su %d tweet di test...\n', num_test);
t_start = tic;

% --- 2. EVALUATION LOOP ---
for i = 1:BATCH_SIZE:num_test
    idx_end = min(i + BATCH_SIZE - 1, num_test);
    x_batch = X_Test(i:idx_end, :);
    y_batch = Y_Test(i:idx_end);
    
    % forward pass without dropout
    [prob, ~] = forward_lstm(x_batch, lstm_net, 0);
    pred = round(prob);
    
    % update metrics and confusion matrix counters
    for k = 1:length(y_batch)
        p = pred(k);
        y = y_batch(k);
        
        if p == y
            correct = correct + 1;
            if y == 1, TP = TP + 1; else, TN = TN + 1; end
        else
            if y == 1, FN = FN + 1; else, FP = FP + 1; end
        end
    end
end

% --- 3. METRICS CALCULATION ---
acc = (correct / num_test) * 100;
prec = TP / (TP + FP + 1e-10);
rec = TP / (TP + FN + 1e-10);
f1 = 2 * (prec * rec) / (prec + rec + 1e-10);

% --- 4. RESULTS OUTPUT ---
fprintf('\n=====================================\n');
fprintf('   RISULTATI FINALI TEST (SGD NESTEROV)\n');
fprintf('=====================================\n');
fprintf('Accuratezza: %.2f%%\n', acc);
fprintf('Tempo Test:  %.2f sec\n', toc(t_start));
fprintf('\nMatrice di Confusione:\n');
fprintf('               Pred 0      Pred 1\n');
fprintf('Reale 0        %-10d  %-10d\n', TN, FP);
fprintf('Reale 1        %-10d  %-10d\n', FN, TP);
fprintf('\nPrecision:   %.4f\n', prec);
fprintf('Recall:      %.4f\n', rec);
fprintf('F1-Score:    %.4f\n', f1);
fprintf('=====================================\n');
