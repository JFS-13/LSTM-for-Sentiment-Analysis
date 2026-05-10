% Forward LSTM

function [prob, cache] = forward_lstm(x_batch, net, dropout_rate)
    % Input arguments.
    %   x_batch: matrix of word indices for the current batch (N x T)
    %   net: structure containing network parameters (Weights and Biases)
    %   dropout_rate (not mandatory): rate for dropout regularization (default 0)
    %
    % Output arguments.
    %   prob: predicted probabilities for each sequence in the batch (N x 1)
    %   cache: structure containing intermediate values needed for backpropagation

    % check input arguments
    if nargin < 3, dropout_rate = 0; end

    % initialization
    [N, T] = size(x_batch); 
    H = size(net.Wf, 2);    
    D = size(net.We, 2);    
    
    % hidden and cell states (3D tensor for batch processing)
    h_states = zeros(N, H, T + 1);
    c_states = zeros(N, H, T + 1);
    
    % gates and dropout cache
    gates_f = zeros(N, H, T); gates_i = zeros(N, H, T);
    gates_c = zeros(N, H, T); gates_o = zeros(N, H, T);
    x_vectors = zeros(N, D, T);
    dropout_masks = zeros(N, H, T);
    
    sig = @(x) 1 ./ (1 + exp(-x));
    
    % recurrence loop
    for t = 1:T
        idx = x_batch(:, t); 
        mask_padding = (idx > 0); 
        
        % embedding look-up with padding handling
        idx_safe = idx; idx_safe(idx == 0) = 1; 
        xt = net.We(idx_safe, :); 
        xt(~mask_padding, :) = 0; 
        
        h_prev = h_states(:, :, t);
        c_prev = c_states(:, :, t);
        
        % gate computations
        f = sig(xt * net.Wf + h_prev * net.Uf + net.bf);
        i = sig(xt * net.Wi + h_prev * net.Ui + net.bi);
        c_tilde = tanh(xt * net.Wc + h_prev * net.Uc + net.bc);
        o = sig(xt * net.Wo + h_prev * net.Uo + net.bo);
        
        % state updates
        c_curr_raw = (f .* c_prev) + (i .* c_tilde);
        h_curr_raw = o .* tanh(c_curr_raw);
        
        % dropout application
        if dropout_rate > 0
            m = (rand(N, H) > dropout_rate);
            scale = 1 / (1 - dropout_rate);
            h_curr_raw = h_curr_raw .* m * scale;
            dropout_masks(:, :, t) = m * scale;
        end
        
        % padding logic: maintain previous state for padded tokens
        h_states(:, :, t+1) = mask_padding .* h_curr_raw + (~mask_padding) .* h_prev;
        c_states(:, :, t+1) = mask_padding .* c_curr_raw + (~mask_padding) .* c_prev;
        
        % cache current step
        gates_f(:,:,t) = f; gates_i(:,:,t) = i;
        gates_c(:,:,t) = c_tilde; gates_o(:,:,t) = o;
        x_vectors(:,:,t) = xt;
    end
    
    % final output layer
    h_final = h_states(:, :, end);
    
    % check for optional dense layer
    if isfield(net, 'W_dense')
        h_dense = tanh(h_final * net.W_dense + net.b_dense);
        logits = (h_dense * net.Why) + net.by;
    else
        logits = (h_final * net.Why) + net.by;
    end
    
    % final activation
    prob = sig(logits);
    
    % pack cache structure
    cache.x_batch = x_batch; cache.h_states = h_states; cache.c_states = c_states;
    cache.f = gates_f; cache.i = gates_i; cache.c = gates_c; cache.o = gates_o;
    cache.x_vectors = x_vectors; cache.prob = prob;
    cache.dropout_masks = dropout_masks; cache.dropout_rate = dropout_rate;
    if isfield(net, 'W_dense'), cache.h_dense = h_dense; end
end
