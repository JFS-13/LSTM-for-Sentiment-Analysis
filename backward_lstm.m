% Backward LSTM

function grads = backward_lstm(x_batch, target, net, cache)
    % Input arguments.
    %   x_batch: matrix of word indices for the current batch (N x T)
    %   target: vector of ground truth labels (N x 1)
    %   net: structure containing network parameters (Weights and Biases)
    %   cache: structure containing cached values from the forward pass, including:
    %       cache.h_states: hidden states over time
    %       cache.c_states: cell states over time
    %       cache.f, cache.i, cache.c, cache.o: gate activations
    %       cache.x_vectors: embedded input vectors
    %       cache.dropout_masks: masks applied during forward pass
    %       cache.prob: predicted probabilities
    %
    % Output arguments.
    %   grads: structure containing the gradients for each parameter in 'net'

    % extraction of dimensions and cache
    [N, T] = size(x_batch);
    h_s = cache.h_states; c_s = cache.c_states;
    f_gate = cache.f; i_gate = cache.i; c_gate = cache.c; o_gate = cache.o;
    x_vecs = cache.x_vectors;
    dropout_masks = cache.dropout_masks;
    use_dropout = (cache.dropout_rate > 0);
    
    [vocab_size, embed_size] = size(net.We);
    hidden_size = size(net.Wf, 2);
    
    % output error (N x 1)
    dy = (cache.prob - target); 
    
    % --- 1. DENSE LAYER / OUTPUT GRADIENTS ---
    if isfield(net, 'W_dense')
        h_dense = cache.h_dense;
        grads.dWhy = (h_dense' * dy);
        grads.dby = sum(dy, 1);
        
        dh_dense = dy * net.Why'; 
        % backpropagation through tanh of dense layer
        d_act = dh_dense .* (1 - h_dense.^2);
        
        grads.dW_dense = (h_s(:,:,end)' * d_act);
        grads.db_dense = sum(d_act, 1);
        dh_next = d_act * net.W_dense'; 
    else
        h_final = h_s(:, :, end);
        grads.dWhy = (h_final' * dy);
        grads.dby = sum(dy, 1);
        dh_next = dy * net.Why';
    end
    
    % --- 2. LSTM PARAMETERS GRADIENTS INITIALIZATION ---
    grads.dWe = zeros(vocab_size, embed_size);
    grads.dWf = zeros(embed_size, hidden_size); grads.dUf = zeros(hidden_size, hidden_size); grads.dbf = zeros(1, hidden_size);
    grads.dWi = zeros(embed_size, hidden_size); grads.dUi = zeros(hidden_size, hidden_size); grads.dbi = zeros(1, hidden_size);
    grads.dWc = zeros(embed_size, hidden_size); grads.dUc = zeros(hidden_size, hidden_size); grads.dbc = zeros(1, hidden_size);
    grads.dWo = zeros(embed_size, hidden_size); grads.dUo = zeros(hidden_size, hidden_size); grads.dbo = zeros(1, hidden_size);
    
    dc_next = zeros(N, hidden_size);
    
    % --- 3. BACKPROPAGATION THROUGH TIME (BPTT) ---
    for t = T:-1:1
        mask_p = (x_batch(:, t) > 0);
        
        % padding logic: skip gradients for padded tokens
        % dh_pass/dc_pass pass the gradient through without modification for padded tokens
        dh_pass = dh_next .* (~mask_p);
        dc_pass = dc_next .* (~mask_p);
        
        % filter dh_next for real tokens
        dh_current = dh_next .* mask_p;
        
        if use_dropout
            dh_current = dh_current .* dropout_masks(:,:,t);
        end
        
        h_prev = h_s(:, :, t); 
        c_curr = c_s(:, :, t+1); 
        c_prev = c_s(:, :, t);
        f = f_gate(:,:,t); i = i_gate(:,:,t); g = c_gate(:,:,t); o = o_gate(:,:,t);
        xt = x_vecs(:,:,t);
        
        tanh_c = tanh(c_curr);
        
        % gates gradients
        do_raw = (dh_current .* tanh_c) .* (o .* (1 - o));
        dc_curr = (dh_current .* o .* (1 - tanh_c.^2)) + dc_next .* mask_p;
        
        dg_raw = (dc_curr .* i) .* (1 - g.^2);
        di_raw = (dc_curr .* g) .* (i .* (1 - i));
        df_raw = (dc_curr .* c_prev) .* (f .* (1 - f));
        
        % parameter accumulation
        grads.dWo = grads.dWo + xt' * do_raw; grads.dUo = grads.dUo + h_prev' * do_raw; grads.dbo = grads.dbo + sum(do_raw, 1);
        grads.dWc = grads.dWc + xt' * dg_raw; grads.dUc = grads.dUc + h_prev' * dg_raw; grads.dbc = grads.dbc + sum(dg_raw, 1);
        grads.dWi = grads.dWi + xt' * di_raw; grads.dUi = grads.dUi + h_prev' * di_raw; grads.dbi = grads.dbi + sum(di_raw, 1);
        grads.dWf = grads.dWf + xt' * df_raw; grads.dUf = grads.dUf + h_prev' * df_raw; grads.dbf = grads.dbf + sum(df_raw, 1);
        
        dxt = df_raw * net.Wf' + di_raw * net.Wi' + dg_raw * net.Wc' + do_raw * net.Wo';
        
        % update embedding gradients
        for n = 1:N
            if mask_p(n)
                idx = x_batch(n, t);
                grads.dWe(idx, :) = grads.dWe(idx, :) + dxt(n, :);
            end
        end
        
        % recurrence for t-1
        dh_next = (df_raw * net.Uf' + di_raw * net.Ui' + dg_raw * net.Uc' + do_raw * net.Uo') + dh_pass;
        dc_next = (dc_curr .* f) + dc_pass;
    end
    
    % --- 4. BATCH NORMALIZATION (Final average) ---
    fn = fieldnames(grads);
    for k = 1:numel(fn)
        grads.(fn{k}) = grads.(fn{k}) / N;
    end
end
