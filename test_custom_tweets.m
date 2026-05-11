% Test Custom Tweets

% This script allows for qualitative evaluation of the trained LSTM models 
% by predicting sentiment scores for custom, user-defined tweets.
% It includes the same preprocessing pipeline (cleaning and stop-word removal) 
% used during training and supports all four optimized models.

clear; clc;

% --- 1. CONFIGURATION ---
% select the model to use for testing: 'adagrad', 'rmsprop', 'adamw', 'sgd'
MODELLO_DA_USARE = 'rmsprop'; 

switch lower(MODELLO_DA_USARE)
    case 'adagrad', file_mat = 'best_lstm_adagrad.mat';
    case 'rmsprop', file_mat = 'best_lstm_rmsprop.mat';
    case 'adamw',   file_mat = 'best_lstm_adamW.mat';
    case 'sgd',     file_mat = 'best_lstm_SGD.mat';
    otherwise,      error('Modello non riconosciuto.');
end

% --- 2. LOADING ---
if isfile('ready_data.mat') && isfile(file_mat)
    load('ready_data.mat', 'wordMap', 'SEQ_LENGTH');
    load(file_mat, 'lstm_net');
else
    error('File necessari non trovati. Controlla ready_data.mat e i modelli .mat');
end

% --- 3. TEST EXAMPLES (Custom Tweets) ---
examples = {
    % POSITIVE
    'Thanks for the amazing support today, really happy with the great result!';
    'I love my new phone, it looks awesome and the camera is so cool!';
    'Having a wonderful time at the beach with my best friends, life is beautiful!';
    'Good morning everyone! Hope you all have a fantastic and productive day.';
    
    % NEGATIVE
    'I hate waiting in line, this service sucks and I feel so tired and sad.';
    'My computer is broken and I lost all my work, this is the worst day ever.';
    'Feeling sick and stuck at home while it is raining outside, so boring.';
    'I missed the bus again, now I will be late for my exam, ugh.';
    
    % NEUTRAL / MIXED
    'Going to work now, see you later for lunch.';
    'The weather is okay today, not too hot but a bit cloudy.';
    'Watching a movie and eating some food, just a quiet night at home.';
    'Thinking about starting a new project next week, we will see how it goes.'
};

% stop words list
stopWords = {'a','an','the','and','or','but','if','because','as','until','while',...
    'of','at','by','for','with','about','against','between','into','through',...
    'during','before','after','above','below','to','from','up','down','in','out',...
    'on','off','over','under','again','further','then','once','here','there',...
    'when','where','why','how','all','any','both','each','few','more','most',...
    'other','some','such','no','nor','not','only','own','same','so','than',...
    'too','very','s','t','can','will','just','don','should','now','d','ll','m','o','re','ve','y',...
    'this','is','it','they','you','are','be','was','were','am','been','has','have','had','do','does','did'};

% --- 4. PRE-PROCESSING AND PREDICTION ---
fprintf('============================================================\n');
fprintf('   TEST SENTIMENT ANALYSIS - Modello: %s\n', upper(MODELLO_DA_USARE));
fprintf('============================================================\n');
fprintf('%-60s | %-10s | %-10s\n', 'Tweet', 'Score', 'Label');
fprintf('------------------------------------------------------------\n');

for i = 1:length(examples)
    tweet = examples{i};
    
    % text cleaning pipeline
    txt = lower(tweet);
    txt = regexprep(txt, 'http\S+', ''); 
    txt = regexprep(txt, '@\S+', ''); 
    txt = regexprep(txt, '([a-z])\1{2,}', '$1'); 
    txt = regexprep(txt, '[^a-z ]', ''); 
    txt = regexprep(txt, '\s+', ' ');
    
    tokens = strsplit(strtrim(txt), ' ');
    tokens = tokens(~cellfun('isempty', tokens));
    
    % remove stop words
    tokens = tokens(~ismember(tokens, stopWords));
    
    % convert to indices
    indices = zeros(1, SEQ_LENGTH);
    count = 0;
    active_tokens = {};
    for w = 1:length(tokens)
        word = tokens{w};
        if isKey(wordMap, word)
            count = count + 1;
            indices(count) = wordMap(word);
            active_tokens{end+1} = word;
        end
        if count >= SEQ_LENGTH, break; end
    end
    
    % forward pass for prediction
    [prob, ~] = forward_lstm(indices, lstm_net, 0);
    
    % score interpretation
    if prob > 0.6
        label = 'POSITIVO';
    elseif prob < 0.4
        label = 'NEGATIVO';
    else
        label = 'NEUTRO/INCERTO';
    end
    
    fprintf('%-60s | %-10.4f | %-10s\n', tweet, prob, label);
end
fprintf('============================================================\n');
