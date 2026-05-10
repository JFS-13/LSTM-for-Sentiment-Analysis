% Train LSTM Adagrad

% This script implements the training loop for the LSTM network using 
% the Adagrad optimization method. It employs adaptive learning rates 
% based on the cumulative sum of squared gradients for each parameter.

clear; clc;

% --- 1. DATA LOADING AND INITIALIZATION ---
disp('Caricamento dati e inizializzazione Adagrad...');
if isfile('ready_data.mat')
    load('ready_data.mat'); 
else
    error('File ready_data.mat non trovato! Esegui load_twitterdata_optimized.m');
end

% hyper-parameters for optimization
BATCH_SIZE = 32;
EPOCHS = 10;            
LEARNING_RATE = 0.01;  
EPSILON = 1e-8;        
CLIP_NORM = 1.0;       
WEIGHT_DECAY = 0.005;  
DROPOUT_RATE = 0.3;    

% load initial weights for consistency
if isfile('init_lstm.mat')
    load('init_lstm.mat');
    fprintf('Rete caricata da init_lstm.mat\n');
else
    error('File init_lstm.mat non trovato! Esegui init_lstm.m');
end

% dimensions and structures
vocab_input_size = VOCAB_SIZE + 2; 
embed_size = 200;      
hidden_size = 128;
dense_size = 128; 

% initialize gradient accumulator
G_state = struct();
fn = fieldnames(lstm_net);
for k = 1:numel(fn)
    G_state.(fn{k}) = zeros(size(lstm_net.(fn{k})));
end

% --- 2. TRAINING LOOP ---
num_train = size(X_Train, 1);
global_iter = 0;
loss_history = [];
val_acc_history = [];

fprintf('Inizio Training Adagrad: %d Epoche\n', EPOCHS);

for epoch = 1:EPOCHS
    total_loss = 0;
    perm = randperm(num_train);
    X_shuffled = X_Train(perm, :);
    Y_shuffled = Y_Train(perm);
    t_start = tic;
    
    for i = 1:BATCH_SIZE:num_train
        global_iter = global_iter + 1;
        idx_end = min(i + BATCH_SIZE - 1, num_train);
        x_batch = X_shuffled(i:idx_end, :);
        y_batch = Y_shuffled(i:idx_end);
        
        % forward pass with dropout
        [prob, cache] = forward_lstm(x_batch, lstm_net, DROPOUT_RATE);
        
        % loss calculation
        batch_loss = -mean(y_batch .* log(prob + 1e-10) + (1 - y_batch) .* log(1 - prob + 1e-10));
        total_loss = total_loss + batch_loss * size(x_batch, 1);
        
        % optimized backward pass
        grads = backward_lstm_optimized(x_batch, y_batch, lstm_net, cache);
        
        % --- ADAGRAD UPDATE STEP ---
        params = fieldnames(lstm_net);
        
        % global gradient clipping
        gnorm_sq = 0;
        for k = 1:numel(params)
            p = params{k};
            gp = grads.(['d' p]);
            gnorm_sq = gnorm_sq + sum(gp(:).^2);
        end
        gnorm = sqrt(gnorm_sq);
        scale = min(1.0, CLIP_NORM / (gnorm + 1e-6));
        
        for k = 1:numel(params)
            p = params{k};
            g = grads.(['d' p]) * scale;
            
            % 1. accumulate squared gradients: G_t = G_t-1 + g^2
            G_state.(p) = G_state.(p) + g.^2;
            
            % 2. calculate adaptive learning rate: lr_t = eta / sqrt(G_t + epsilon)
            adaptive_lr = LEARNING_RATE ./ (sqrt(G_state.(p)) + EPSILON);
            
            % decoupled weight decay (applied before update)
            if ~contains(p, 'b')
                lstm_net.(p) = lstm_net.(p) * (1 - LEARNING_RATE * WEIGHT_DECAY);
            end
            
            % 3. parameter update: theta = theta - lr_t * g
            lstm_net.(p) = lstm_net.(p) - adaptive_lr .* g;
        end
        
        if mod(global_iter, 500) == 0
            fprintf('  Ep %d | Step %d | Loss: %.4f | GNorm: %.2f\n', epoch, global_iter, total_loss/idx_end, gnorm);
        end
    end
    
    % --- END OF EPOCH: VALIDATION ---
    epoch_loss = total_loss / num_train;
    loss_history(end+1) = epoch_loss;
    correct_val = 0;
    num_val = size(X_Val, 1);
    for v = 1:BATCH_SIZE:num_val
        v_end = min(v + BATCH_SIZE - 1, num_val);
        [p_v, ~] = forward_lstm(X_Val(v:v_end, :), lstm_net, 0); 
        correct_val = correct_val + sum(round(p_v) == Y_Val(v:v_end));
    end
    val_acc = (correct_val / num_val) * 100;
    val_acc_history(end+1) = val_acc;
    
    fprintf('>> FINE EPOCA %d: Loss=%.4f, ValAcc=%.2f%% | Tempo: %.1fs\n', epoch, epoch_loss, val_acc, toc(t_start));
    
    if epoch == 1 || val_acc > max(val_acc_history(1:end-1))
        save('best_lstm_adagrad.mat', 'lstm_net', 'loss_history', 'val_acc_history', 'VOCAB_SIZE', 'SEQ_LENGTH');
    end
end
