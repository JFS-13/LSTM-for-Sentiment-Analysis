% K-fold Grid Search

% This script implements a systematic Grid Search with K-Fold Cross-Validation 
% to optimize both the Learning Rate (LR) and the Regularization parameter (lambda).
% It uses PARFOR to parallelize the search across different hyper-parameter 
% combinations, significantly reducing execution time.

clear; clc; close all;

% --- 1. GRID SEARCH CONFIGURATION ---
% select the optimizer to test: 'sgd', 'adagrad', 'rmsprop', 'adamw'
OPTIMIZER = 'sgd'; 

% hyper-parameter grid
LR_VALUES = [1e-2, 1e-3, 5e-4];             
LAMBDA_VALUES = [1e-4, 1e-3, 5e-3, 1e-2];   

K_FOLDS = 3;            
init_net_file = 'init_lstm.mat';

% data loading
if ~isfile('ready_data.mat')
    error('File ready_data.mat non trovato! Esegui load_twitterdata.m');
end
load('ready_data.mat'); 

% merge Train + Val for Cross-Validation
X_CV = [X_Train; X_Val];
Y_CV = [Y_Train; Y_Val];

num_samples = size(X_CV, 1);
indices = randperm(num_samples);
fold_size = floor(num_samples / K_FOLDS);

% Generate list of all combinations for parfor
[LR_grid, LAMBDA_grid] = meshgrid(LR_VALUES, LAMBDA_VALUES);
combinations = [LR_grid(:), LAMBDA_grid(:)];
num_comb = size(combinations, 1);
results_acc = zeros(num_comb, 1);

fprintf('=========================================\n');
fprintf('PARALLEL GRID SEARCH: %s\n', upper(OPTIMIZER));
fprintf('=========================================\n');
fprintf('Inizio %d-Fold Cross Validation su %d campioni.\n', K_FOLDS, num_samples);
fprintf('Combinazioni totali da testare: %d\n', num_comb);

% Check for Parallel Pool
p = gcp('nocreate');
if isempty(p)
    fprintf('Attivazione Parallel Pool...\n');
    parpool; % Start default parallel pool
end

% --- 2. PARALLEL GRID SEARCH LOOP ---
total_timer = tic;

parfor i = 1:num_comb
    cur_lr = combinations(i, 1);
    cur_lambda = combinations(i, 2);
    
    % A local variable is used to store fold accuracies
    local_fold_accs = zeros(K_FOLDS, 1);
    
    for k = 1:K_FOLDS
        % rotational split for CV
        val_start = (k-1)*fold_size + 1;
        val_end   = k*fold_size;
        
        idx_v = indices(val_start:val_end);
        idx_t = indices; 
        idx_t(val_start:val_end) = []; 
        
        X_T_Fold = X_CV(idx_t, :);
        Y_T_Fold = Y_CV(idx_t);
        X_V_Fold = X_CV(idx_v, :);
        Y_V_Fold = Y_CV(idx_v);
        
        % call generic training function
        % best_val_acc = train_fold_generic(...)
        local_fold_accs(k) = train_fold_generic(X_T_Fold, Y_T_Fold, X_V_Fold, Y_V_Fold, cur_lambda, cur_lr, init_net_file, OPTIMIZER);
    end
    
    results_acc(i) = mean(local_fold_accs);
    % Synchronized log (will print when worker finishes combination)
    fprintf('Combinazione %d/%d completata [LR: %.1e, L: %.1e] -> Acc: %.2f%%\n', ...
            i, num_comb, cur_lr, cur_lambda, results_acc(i));
end

% --- 3. RESULTS ANALYSIS ---
results_grid = [combinations, results_acc];
[best_acc, best_idx] = max(results_acc);
best_lr = results_grid(best_idx, 1);
best_lambda = results_grid(best_idx, 2);

fprintf('\n=========================================\n');
fprintf('RISULTATI FINALI GRID SEARCH (%s)\n', upper(OPTIMIZER));
fprintf('=========================================\n');
res_table = array2table(results_grid, 'VariableNames', {'LR', 'Lambda', 'Mean_Val_Accuracy'});
disp(res_table);

fprintf('\nVINCITORE:\n');
fprintf('Best Learning Rate: %.1e\n', best_lr);
fprintf('Best Lambda (Weight Decay):   %.1e\n', best_lambda);
fprintf('Validation Acc:     %.2f%%\n', best_acc);

toc(total_timer);