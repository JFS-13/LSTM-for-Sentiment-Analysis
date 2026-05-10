% Initialize LSTM

% This script initializes the LSTM network architecture and weights.
% It defines the hyper-parameters (embedding size, hidden size, etc.) 
% and uses Xavier initialization for all weight matrices to ensure 
% stable training across different optimizers.

% load data dimensions
if isfile('ready_data.mat')
    load('ready_data.mat', 'VOCAB_SIZE', 'SEQ_LENGTH');
else
    error('ready_data.mat non trovato. Esegui prima load_twitterdata.m');
end

% hyper-parameters configuration
input_size  = VOCAB_SIZE + 2; 
embed_size  = 200;  
hidden_size = 128;  
dense_size  = 128;
output_size = 1;   

fprintf('Architettura LSTM:\n');
fprintf('- Embedding: %d\n', embed_size);
fprintf('- Hidden/Cell Size: %d\n', hidden_size);
fprintf('- Dense Layer: %d\n', dense_size);

% weight initialization (Xavier/Glorot)
% helper function for Xavier initialization
xavier = @(n_in, n_out) (rand(n_in, n_out) * 2 * (sqrt(6)/sqrt(n_in+n_out))) - (sqrt(6)/sqrt(n_in+n_out));

% --- A. EMBEDDING LAYER ---
lstm_net.We = xavier(input_size, embed_size);

% --- B. LSTM GATES ---
lstm_net.Wf = xavier(embed_size, hidden_size); lstm_net.Uf = xavier(hidden_size, hidden_size); lstm_net.bf = ones(1, hidden_size);
lstm_net.Wi = xavier(embed_size, hidden_size); lstm_net.Ui = xavier(hidden_size, hidden_size); lstm_net.bi = zeros(1, hidden_size);
lstm_net.Wc = xavier(embed_size, hidden_size); lstm_net.Uc = xavier(hidden_size, hidden_size); lstm_net.bc = zeros(1, hidden_size);
lstm_net.Wo = xavier(embed_size, hidden_size); lstm_net.Uo = xavier(hidden_size, hidden_size); lstm_net.bo = zeros(1, hidden_size);

% --- C. INTERMEDIATE DENSE LAYER ---
lstm_net.W_dense = xavier(hidden_size, dense_size);
lstm_net.b_dense = zeros(1, dense_size);

% --- D. OUTPUT LAYER ---
lstm_net.Why = xavier(dense_size, output_size);
lstm_net.by  = zeros(1, output_size);

% save initialized network
disp('Rete LSTM inizializzata con pesi Xavier.');
save('init_lstm.mat', 'lstm_net');
