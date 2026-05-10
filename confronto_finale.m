% Final Comparison

% This script performs a comparative evaluation of different optimization 
% methods (SGD Nesterov, Adagrad, RMSprop, AdamW) applied to the LSTM model.
% It calculates Accuracy and F1-Score on the test set and generates 
% comparative plots for Training Loss and Validation Accuracy.

clear; clc; close all;

% configuration
metodi = {'SGD_momentum', 'adagrad', 'rmsprop', 'adamW_optimized'};
nomi_label = {'SGD Nesterov', 'Adagrad', 'RMSprop', 'Adam'};
colori = {'r', 'g', 'b', 'm'};
stili = {'-o', '-s', '-^', '-d'}; 

if ~isfile('ready_data.mat')
    error('File ready_data.mat non trovato!');
end
load('ready_data.mat', 'X_Test', 'Y_Test');

results = struct();
num_test = size(X_Test, 1);
BATCH_SIZE = 64;

% evaluation loop
fprintf('Inizio valutazione comparativa sui %d tweet di test...\n', num_test);
fprintf('%-25s | %-10s | %-10s | %-10s\n', 'Metodo', 'Accur.', 'F1-Score', 'Tempo');
fprintf('----------------------------------------------------------------------\n');

figure('Name', 'Confronto Ottimizzatori', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 500]);

for i = 1:length(metodi)
    file_mat = sprintf('best_lstm_%s.mat', metodi{i});
    
    if isfile(file_mat)
        data = load(file_mat);
        
        % test evaluation metrics
        correct = 0; TP = 0; TN = 0; FP = 0; FN = 0;
        t_start = tic;
        
        for b = 1:BATCH_SIZE:num_test
            idx_end = min(b + BATCH_SIZE - 1, num_test);
            [prob, ~] = forward_lstm(X_Test(b:idx_end, :), data.lstm_net, 0);
            pred = round(prob);
            y_true = Y_Test(b:idx_end);
            
            correct = correct + sum(pred == y_true);
            TP = TP + sum((pred == 1) & (y_true == 1));
            TN = TN + sum((pred == 0) & (y_true == 0));
            FP = FP + sum((pred == 1) & (y_true == 0));
            FN = FN + sum((pred == 0) & (y_true == 1));
        end
        
        t_eval = toc(t_start);
        acc = (correct / num_test) * 100;
        prec = TP / (TP + FP + 1e-10);
        rec = TP / (TP + FN + 1e-10);
        f1 = 2 * (prec * rec) / (prec + rec + 1e-10);
        
        fprintf('%-25s | %-9.2f%% | %-10.4f | %-10.2fs\n', nomi_label{i}, acc, f1, t_eval);
        
        % plotting loss history
        subplot(1,2,1);
        plot(data.loss_history, stili{i}, 'Color', colori{i}, 'LineWidth', 2, 'MarkerSize', 5, 'DisplayName', nomi_label{i});
        hold on;
        
        % plotting validation accuracy history
        subplot(1,2,2);
        plot(data.val_acc_history, stili{i}, 'Color', colori{i}, 'LineWidth', 2, 'MarkerSize', 5, 'DisplayName', nomi_label{i});
        hold on;
    else
        fprintf('%-25s | FILE MANCANTE (%s)\n', nomi_label{i}, file_mat);
    end
end

% plot refinement
subplot(1,2,1);
title('Confronto Loss (Caso Non Convesso)');
xlabel('Epoca'); ylabel('Mean Squared Error / Log-Loss');
legend('Location', 'northeast'); grid on;

subplot(1,2,2);
title('Confronto Accuratezza Validazione');
xlabel('Epoca'); ylabel('Accuratezza (%)');
legend('Location', 'southeast'); grid on;

fprintf('----------------------------------------------------------------------\n');
disp('Confronto completato. I grafici mostrano la stabilità della convergenza.');
