% Train Fold Generic

function best_val_acc = train_fold_generic(X_T, Y_T, X_V, Y_V, lambda_val, lr_val, init_net_file, optimizer_type)
    % Input arguments.
    %   X_T: Training data for the current fold (N_train x T)
    %   Y_T: Training labels for the current fold (N_train x 1)
    %   X_V: Validation data for the current fold (N_val x T)
    %   Y_V: Validation labels for the current fold (N_val x 1)
    %   lambda_val: Regularization parameter (weight decay)
    %   lr_val: Initial Learning Rate for the optimizer
    %   init_net_file: Path to the .mat file containing the initial Xavier weights
    %   optimizer_type: String specifying the optimizer ('nesterov', 'adagrad', 'rmsprop', 'adamw')
    %
    % Output arguments.
    %   best_val_acc: Best validation accuracy achieved during the training of the fold

    % load initial master network weights
    loaded = load(init_net_file);
    net = loaded.lstm_net;
    
    % --- BASE HYPER-PARAMETERS ---
    BATCH_SIZE = 32;
    EPOCHS = 3;            % Reduced to 3 for faster Grid Search execution
    CLIP_NORM = 1.0;       
    DROPOUT_RATE = 0.3;    
    EPSILON = 1e-8;

    % initialize optimizer-specific states
    params = fieldnames(net);
    state1 = struct(); % v for nesterov/adamw, G for adagrad/rmsprop
    state2 = struct(); % v_t for adamw
    for k = 1:numel(params)
        state1.(params{k}) = zeros(size(net.(params{k})));
        state2.(params{k}) = zeros(size(net.(params{k})));
    end

    num_train = size(X_T, 1);
    num_val = size(X_V, 1);
    val_acc_history = [];
    global_iter = 0;

    % training loop
    for epoch = 1:EPOCHS
        perm = randperm(num_train);
        X_shuffled = X_T(perm, :);
        Y_shuffled = Y_T(perm);
        
        for i = 1:BATCH_SIZE:num_train
            global_iter = global_iter + 1;
            idx_end = min(i + BATCH_SIZE - 1, num_train);
            
            x_batch = X_shuffled(i:idx_end, :);
            y_batch = Y_shuffled(i:idx_end);
            
            % forward pass with dropout
            [~, cache] = forward_lstm(x_batch, net, DROPOUT_RATE);
            
            % optimized backward pass
            grads = backward_lstm_optimized(x_batch, y_batch, net, cache);
            
            % 1. Global Gradient Clipping (Slide 82)
            gnorm_sq = 0;
            for k = 1:numel(params)
                gp = grads.(['d' params{k}]);
                gnorm_sq = gnorm_sq + sum(gp(:).^2);
            end
            gnorm = sqrt(gnorm_sq);
            scale = min(1.0, CLIP_NORM / (gnorm + 1e-6));
            
            % 2. Update Logic (Slide 80-84)
            for k = 1:numel(params)
                p = params{k};
                g = grads.(['d' p]) * scale;
                
                switch lower(optimizer_type)
                    case 'nesterov'
                        % Robbins-Monro schedule (Slide 80)
                        lr = lr_val / (1 + 1e-4 * global_iter);
                        MOMENTUM = 0.95;
                        state1.(p) = MOMENTUM * state1.(p) + g;
                        update_dir = g + MOMENTUM * state1.(p);
                        
                        if ~contains(p, 'b')
                            net.(p) = net.(p) - lr * update_dir - (lr * lambda_val * net.(p));
                        else
                            net.(p) = net.(p) - lr * update_dir;
                        end

                    case 'adagrad'
                        % cumulative diagonal scaling (Slide 84)
                        state1.(p) = state1.(p) + g.^2;
                        adaptive_lr = lr_val ./ (sqrt(state1.(p)) + EPSILON);
                        
                        if ~contains(p, 'b')
                            net.(p) = net.(p) * (1 - lr_val * lambda_val);
                        end
                        net.(p) = net.(p) - adaptive_lr .* g;

                    case 'rmsprop'
                        % moving average scaling (Slide 84)
                        GAMMA = 0.9;
                        state1.(p) = GAMMA * state1.(p) + (1 - GAMMA) * g.^2;
                        adaptive_lr = lr_val ./ (sqrt(state1.(p)) + EPSILON);
                        
                        if ~contains(p, 'b')
                            net.(p) = net.(p) * (1 - lr_val * lambda_val);
                        end
                        net.(p) = net.(p) - adaptive_lr .* g;

                    case 'adamw'
                        % Adam with bias correction and decoupled weight decay (Slide 84)
                        lr = lr_val / (1 + 1e-4 * global_iter);
                        beta1 = 0.9; beta2 = 0.999;
                        
                        state1.(p) = beta1 * state1.(p) + (1 - beta1) * g;
                        state2.(p) = beta2 * state2.(p) + (1 - beta2) * (g.^2);
                        
                        m_hat = state1.(p) / (1 - beta1^global_iter);
                        v_hat = state2.(p) / (1 - beta2^global_iter);
                        
                        if ~contains(p, 'b')
                            net.(p) = net.(p) - (lr * lambda_val * net.(p));
                        end
                        net.(p) = net.(p) - lr * (m_hat ./ (sqrt(v_hat) + EPSILON));
                end
            end
        end
        
        % end of epoch validation
        correct_val = 0;
        for v = 1:BATCH_SIZE:num_val
            v_end = min(v + BATCH_SIZE - 1, num_val);
            [p_v, ~] = forward_lstm(X_V(v:v_end, :), net, 0); 
            correct_val = correct_val + sum(round(p_v) == Y_V(v:v_end));
        end
        val_acc_history(end+1) = (correct_val / num_val) * 100;
        fprintf('.'); 
    end
    
    best_val_acc = max(val_acc_history);
    fprintf(' Done. Best Acc: %.2f%%\n', best_val_acc);
end
