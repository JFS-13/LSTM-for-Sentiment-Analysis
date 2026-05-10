% Load Twitter Data

% This script handles the entire data preprocessing pipeline for the 
% Twitter sentiment analysis project. It performs text cleaning 
% (URL/mention removal, lowercasing), tokenization, stop-word filtering, 
% dataset splitting (70% train, 15% val, 15% test), and vocabulary 
% construction. The final output is 'ready_data.mat', which contains 
% vectorized matrices suitable for LSTM training.

clear; clc; close all;

% --- 1. CONFIGURATION AND LOADING ---
baseDir = 'TwitterParsed';
categories = {'0', '1'}; 
MAX_FILES = 150000; 

% stop words list
stopWords = {'a','an','the','and','or','but','if','because','as','until','while',...
    'of','at','by','for','with','about','against','between','into','through',...
    'during','before','after','above','below','to','from','up','down','in','out',...
    'on','off','over','under','again','further','then','once','here','there',...
    'when','where','why','how','all','any','both','each','few','more','most',...
    'other','some','such','no','nor','not','only','own','same','so','than',...
    'too','very','s','t','can','will','just','don','should','now','d','ll','m','o','re','ve','y'};

allTokenizedDocs = cell(MAX_FILES, 1); 
allLabels = zeros(MAX_FILES, 1);
docLengths = zeros(MAX_FILES, 1);

cnt = 0;
hWait = waitbar(0, 'Caricamento, pulizia e tokenizzazione...');

for c = 1:length(categories)
    catName = categories{c};
    labelVal = str2double(catName);
    folderPath = fullfile(baseDir, catName);
    files = dir(fullfile(folderPath, '*.txt'));
    
    for k = 1:length(files)
        cnt = cnt + 1;
        filePath = fullfile(folderPath, files(k).name);
        txt = fileread(filePath);
        
        % text cleaning pipeline
        txt = lower(txt);
        txt = regexprep(txt, 'http\S+', ''); 
        txt = regexprep(txt, '@\S+', ''); 
        % reduce repeated letters (e.g., loooove -> love)
        txt = regexprep(txt, '([a-z])\1{2,}', '$1');
        txt = regexprep(txt, '[^a-z ]', ''); 
        txt = regexprep(txt, '\s+', ' ');
        
        tokens = strsplit(strtrim(txt));
        tokens = tokens(~cellfun('isempty', tokens));
        
        % stop words removal
        tokens = tokens(~ismember(tokens, stopWords));

        if isempty(tokens)
            len = 0;
        else
            len = length(tokens);
        end
        
        allTokenizedDocs{cnt} = tokens;
        allLabels(cnt) = labelVal;
        docLengths(cnt) = len;
        
        if mod(cnt, 5000) == 0
            waitbar(cnt/MAX_FILES, hWait, sprintf('Processati %d file...', cnt));
        end
    end
end
close(hWait);

allTokenizedDocs = allTokenizedDocs(1:cnt);
allLabels = allLabels(1:cnt);
docLengths = docLengths(1:cnt);
numTotal = cnt;

% --- 2. SEQUENCE LENGTH ANALYSIS ---
p95 = prctile(docLengths, 95);
SEQ_LENGTH = ceil(p95) + 2; 

% --- 3. DATASET SPLIT (70-15-15) ---
rng(42); 
perm = randperm(numTotal);
idxSplit1 = floor(0.70 * numTotal);
idxSplit2 = floor(0.85 * numTotal);

docsTrain = allTokenizedDocs(perm(1:idxSplit1)); Y_Train = allLabels(perm(1:idxSplit1));
docsVal   = allTokenizedDocs(perm(idxSplit1+1:idxSplit2));   Y_Val   = allLabels(perm(idxSplit1+1:idxSplit2));
docsTest  = allTokenizedDocs(perm(idxSplit2+1:end));  Y_Test  = allLabels(perm(idxSplit2+1:end));

% --- 4. VOCABULARY CONSTRUCTION (Top 10000 words) ---
VOCAB_SIZE = 10000; 
allTokens = [docsTrain{:}]; 
[uniqueWords, ~, ic] = unique(allTokens);
counts = histcounts(ic, 'BinMethod', 'integers')';
[~, sortIdx] = sort(counts, 'descend');
sortedWords = uniqueWords(sortIdx);

topWords = sortedWords(1:min(VOCAB_SIZE, length(sortedWords)));
wordMap = containers.Map(topWords, 2:(length(topWords)+1));

% --- 5. VECTORIZATION ---
X_Train = tokensToMatrix(docsTrain, wordMap, SEQ_LENGTH);
X_Val   = tokensToMatrix(docsVal, wordMap, SEQ_LENGTH);
X_Test  = tokensToMatrix(docsTest, wordMap, SEQ_LENGTH);

save('ready_data.mat', 'X_Train', 'Y_Train', 'X_Val', 'Y_Val', 'X_Test', 'Y_Test', 'wordMap', 'VOCAB_SIZE', 'SEQ_LENGTH');
disp('DONE. Dataset ottimizzato con Stop Words e Vocabolario 10k.');

% helper function for vectorization
function X = tokensToMatrix(docsCells, map, seqLen)
    numDocs = length(docsCells);
    X = zeros(numDocs, seqLen);
    for i = 1:numDocs
        tokens = docsCells{i};
        L = min(length(tokens), seqLen);
        for t = 1:L
            word = tokens{t};
            if isKey(map, word)
                X(i,t) = map(word);
            else
                X(i,t) = 1; % <UNK>
            end
        end
    end
end
