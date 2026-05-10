% Check Vocabulary

% This script displays the words stored in the 'wordMap' and generates
% a text file 'vocabulary_full.txt' containing the entire vocabulary.
% It is used for diagnostic purposes and to identify words for manual testing.

% check input data
if isfile('ready_data.mat')
    load('ready_data.mat', 'wordMap');
    
    % extract and sort vocabulary
    allWords = keys(wordMap);
    allIndices = values(wordMap);
    [~, sortIdx] = sort(cell2mat(allIndices));
    sortedWords = allWords(sortIdx);
    
    numWords = length(sortedWords);
    fprintf('=== Vocabolario Completo (%d parole) ===\n', numWords);
    fprintf('1. Le prime 100 parole sono visualizzate qui sotto.\n');
    fprintf('2. L''intero vocabolario è stato salvato in ''vocabulary_full.txt''.\n');
    fprintf('3. Puoi esplorarlo in Matlab con il comando: disp(sortedWords'')\n\n');

    % preview top 100 words
    for i = 1:min(100, numWords)
        fprintf('%-15s', sortedWords{i});
        if mod(i, 5) == 0, fprintf('\n'); end
    end
    fprintf('\n');

    % save vocabulary to file
    fileID = fopen('vocabulary_full.txt', 'w');
    for i = 1:numWords
        fprintf(fileID, '%d: %s\n', i+1, sortedWords{i}); % i+1 because index 1 is reserved for UNK
    end
    fclose(fileID);
    
    fprintf('\nFile ''vocabulary_full.txt'' generato con successo.\n');
    
    % usage suggestions
    fprintf('\nPer cercare una parola specifica, usa:\n');
    fprintf('find(strcmp(sortedWords, ''tuaparola''))\n');
    
else
    error('ready_data.mat non trovato. Esegui prima load_twitterdata.m');
end
