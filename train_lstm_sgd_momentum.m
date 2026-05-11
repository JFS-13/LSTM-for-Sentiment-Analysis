% LSTM Train SGD Momentum

% This script implements the training loop for the LSTM network using 
% Stochastic Gradient Descent (SGD) with Nesterov Momentum. 
% It employs a diminishing stepsize (Robbins-Monro schedule) and 
% global gradient clipping to handle the non-convex nature of the loss surface.

clear; clc;

% --- 1. DATA LOADING AND INITIALIZATION ---
disp('Caricamento dati e inizializzazione SGD Nesterov...');
if isfile('ready_data.mat')
    load('ready_data.mat'); 
else
    error('File ready_data.mat non trovato! Esegui load_twitterdata.m');
end

% hyper-parameters for optimization
BATCH_SIZE = 32;
EPOCHS = 10;            
INIT_LR = 0.01;         
CLIP_NORM = 1.0;       
WEIGHT_DECAY = 0.001; 
DROPOUT_RATE = 0.3;    
MOMENTUM = 0.95;       

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

% momentum velocity initialization
v_state = struct();
fn = fieldnames(lstm_net);
for k = 1:numel(fn)
    v_state.(fn{k}) = zeros(size(lstm_net.(fn{k})));
end

% --- 2. TRAINING LOOP ---
num_train = size(X_Train, 1);
total_steps = ceil(num_train / BATCH_SIZE) * EPOCHS;
global_iter = 0;

loss_history = [];
val_acc_history = [];

fprintf('Inizio Training SGD Nesterov: %d Epoche\n', EPOCHS);

for epoch = 1:EPOCHS
    total_loss = 0;
    perm = randperm(num_train);
    X_shuffled = X_Train(perm, :);
    Y_shuffled = Y_Train(perm);
    
    t_start = tic;
    
    for i = 1:BATCH_SIZE:num_train
        global_iter = global_iter + 1;
        idx_end = min(i + BATCH_SIZE - 1, num_train);
        this_batch_size = idx_end - i + 1;
        
        x_batch = X_shuffled(i:idx_end, :);
        y_batch = Y_shuffled(i:idx_end);
        
        % A. LR SCHEDULER (Diminishing Stepsize - Robbins-Monro)
        decay_rate = 1e-4; 
        lr = INIT_LR / (1 + decay_rate * global_iter);
        
        % B. FORWARD PASS
        [prob, cache] = forward_lstm(x_batch, lstm_net, DROPOUT_RATE);
        
        % loss calculation
        batch_loss = -mean(y_batch .* log(prob + 1e-10) + (1 - y_batch) .* log(1 - prob + 1e-10));
        total_loss = total_loss + batch_loss * this_batch_size;
        
        % C. BACKWARD PASS
        grads = backward_lstm(x_batch, y_batch, lstm_net, cache);
        
        % D. NESTEROV UPDATE STEP
        params = fieldnames(lstm_net);
        
        % 1. Global Gradient Norm Calculation
        gnorm_sq = 0;
        for k = 1:numel(params)
            p = params{k};
            gp = grads.(['d' p]);
            gnorm_sq = gnorm_sq + sum(gp(:).^2);
        end
        gnorm = sqrt(gnorm_sq);
        
        scale = min(1.0, CLIP_NORM / (gnorm + 1e-6));
        
        % 2. Parameter Update with Nesterov Logic
        for k = 1:numel(params)
            p = params{k};
            g = grads.(['d' p]) * scale;
            
            % momentum velocity update
            v_state.(p) = MOMENTUM * v_state.(p) + g;
            
            % Nesterov: update using future velocity
            update_dir = g + MOMENTUM * v_state.(p);
            
            if contains(p, 'b')
                lstm_net.(p) = lstm_net.(p) - lr * update_dir;
            else
                % Decoupled Weight Decay
                lstm_net.(p) = lstm_net.(p) - lr * update_dir - (lr * WEIGHT_DECAY * lstm_net.(p));
            end
        end
        
        if mod(global_iter, 500) == 0
            fprintf('  Ep %d | Step %d | Loss: %.4f | LR: %.6f | GNorm: %.2f\n', ...
                epoch, global_iter, total_loss/idx_end, lr, gnorm);
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
    
    fprintf('>> FINE EPOCA %d: Loss=%.4f, ValAcc=%.2f%% | Tempo: %.1fs\n', ...
        epoch, epoch_loss, val_acc, toc(t_start));
        
    if epoch == 1 || val_acc > max(val_acc_history(1:end-1))
        save('best_lstm_SGD.mat', 'lstm_net', 'loss_history', 'val_acc_history', 'VOCAB_SIZE', 'SEQ_LENGTH');
    end
end

% visualization
figure;
subplot(2,1,1); plot(loss_history, 'r-o'); title('Training Loss SGD');
subplot(2,1,2); plot(val_acc_history, 'b-s'); title('Validation Accuracy');
grid on;
