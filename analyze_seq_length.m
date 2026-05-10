% Analyze sequence length

% This script analyzes the distribution of tweet lengths within the 
% TwitterParsed dataset. The goal is to identify a suitable SEQ_LENGTH 
% for the LSTM network by calculating percentiles (50th, 90th, 95th, 99th) 
% of the tokenized and cleaned text.

clear; clc; close all;

% configuration
baseDir = 'TwitterParsed';
categories = {'0', '1'}; 

% initialization
allLengths = [];
fprintf('Analisi lunghezza tweet (Tokenizzazione semplice)...\n');
hWait = waitbar(0, 'Lettura file...');
totalFilesProcessed = 0;

% process files
for c = 1:length(categories)
    folderPath = fullfile(baseDir, categories{c});
    files = dir(fullfile(folderPath, '*.txt'));
    
    for k = 1:length(files)
        totalFilesProcessed = totalFilesProcessed + 1;
        filePath = fullfile(folderPath, files(k).name);
        
        % read content
        txt = fileread(filePath);
        
        % text cleaning (consistent with load_twitterdata_optimized)
        txt = lower(txt);
        txt = regexprep(txt, 'http\S+', ''); 
        txt = regexprep(txt, '@\S+', ''); 
        txt = regexprep(txt, '[^a-z ]', ''); 
        
        % tokenization
        tokens = strsplit(strtrim(txt));
        
        % count tokens
        if length(tokens) == 1 && isempty(tokens{1})
            thisLen = 0;
        else
            thisLen = length(tokens);
        end
        
        allLengths(end+1) = thisLen;
        
        if mod(totalFilesProcessed, 5000) == 0
            waitbar(k/length(files), hWait, sprintf('Analizzati %d file...', totalFilesProcessed));
        end
    end
end
close(hWait);

% statistics and plotting
figure('Color','w');

% histogram
histogram(allLengths, 'BinWidth', 1, 'FaceColor', [0 0.4470 0.7410]);
title('Distribuzione Lunghezza Tweet');
xlabel('Numero di Token per Tweet');
ylabel('Frequenza');
grid on;
xlim([0, 80]); 

% calculate percentiles
p50 = prctile(allLengths, 50);
p90 = prctile(allLengths, 90);
p95 = prctile(allLengths, 95);
p99 = prctile(allLengths, 99);

% draw vertical lines
hold on;
xline(p50, '--k', ['Media (50%): ' num2str(p50)], 'LineWidth', 1.5, 'LabelVerticalAlignment', 'top');
xline(p90, '--m', ['90%: ' num2str(p90)], 'LineWidth', 1.5);
xline(p95, '-.r', ['95%: ' num2str(p95)], 'LineWidth', 2);

% display results
fprintf('\n=== RISULTATI ANALISI ===\n');
fprintf('Tweet totali: %d\n', length(allLengths));
fprintf('Lunghezza Media: %.1f token\n', mean(allLengths));
fprintf('Lunghezza Max:   %d token\n', max(allLengths));
fprintf('\nPercentili (Copertura):\n');
fprintf('  50%% dei tweet ha <= %d token\n', ceil(p50));
fprintf('  90%% dei tweet ha <= %d token\n', ceil(p90));
fprintf('  95%% dei tweet ha <= %d token\n', ceil(p95));
fprintf('  99%% dei tweet ha <= %d token\n', ceil(p99));
fprintf('=========================\n');
