% Train LSTM AdamW

% This script implements an optimized training loop for the LSTM network 
% using the AdamW optimization method. It includes bias correction for 
% moment estimates and decoupled weight decay, ensuring stable convergence.

clear; clc;

% --- 1. DATA LOADING AND INITIALIZATION ---
disp('Caricamento dati e inizializzazione AdamW...');
if isfile('ready_data.mat')
    load('ready_data.mat'); 
else
    error('File ready_data.mat non trovato! Esegui load_twitterdata.m');
end

% hyper-parameters for optimization
BATCH_SIZE = 32;
EPOCHS = 10;           
INIT_LR = 0.001;       
CLIP_NORM = 1.0;       
WEIGHT_DECAY = 0.001;   
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

% Adam moment initialization
m_state = struct(); v_state = struct();
fn = fieldnames(lstm_net);
for k = 1:numel(fn)
    m_state.(fn{k}) = zeros(size(lstm_net.(fn{k})));
    v_state.(fn{k}) = zeros(size(lstm_net.(fn{k})));
end

% --- 2. TRAINING LOOP ---
num_train = size(X_Train, 1);
total_steps = ceil(num_train / BATCH_SIZE) * EPOCHS;
global_iter = 0;

loss_history = [];
val_acc_history = [];

fprintf('Inizio Training AdamW: %d Epoche\n', EPOCHS);

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
        
        % --- ADAMW UPDATE STEP ---
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
        
        beta1 = 0.9;
        beta2 = 0.999;
        epsilon = 1e-8;
        
        for k = 1:numel(params)
            p = params{k};
            g = grads.(['d' p]) * scale;
            
            % 1. first moment estimate (m_t)
            m_state.(p) = beta1 * m_state.(p) + (1 - beta1) * g;
            
            % 2. second moment estimate (v_t)
            v_state.(p) = beta2 * v_state.(p) + (1 - beta2) * (g.^2);
            
            % 3. bias correction
            m_hat = m_state.(p) / (1 - beta1^global_iter);
            v_hat = v_state.(p) / (1 - beta2^global_iter);
            
            % 4. update calculation
            update_step = lr * (m_hat ./ (sqrt(v_hat) + epsilon));
            
            % decoupled weight decay
            if ~contains(p, 'b')
                lstm_net.(p) = lstm_net.(p) - (lr * WEIGHT_DECAY * lstm_net.(p));
            end
            
            % 5. parameter update
            lstm_net.(p) = lstm_net.(p) - update_step;
        end
        
        if mod(global_iter, 500) == 0
            fprintf('  Ep %d | Step %d | Loss: %.4f | LR: %.6f\n', ...
                epoch, global_iter, total_loss/idx_end, lr);
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
        save('best_lstm_adamW.mat', 'lstm_net', 'loss_history', 'val_acc_history', 'VOCAB_SIZE', 'SEQ_LENGTH');
    end
end

% visualization
figure;
subplot(2,1,1); plot(loss_history, 'm-o'); title('Training Loss (AdamW Optimized)');
subplot(2,1,2); plot(val_acc_history, 'g-s'); title('Validation Accuracy');
grid on;
