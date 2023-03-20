clear all;

% read data
load('A');
load BANDS;

% type = 5;
% c = 4;
% A = A(BANDS, [1:c]);
% [mixed, abf] = getSynData(A, 7, 0);
% [M, N, D] = size(mixed);
% mixed = reshape(mixed, M * N, D);

% add noise
% SNR = 20; %dB
% variance = sum(mixed(:) .^ 2) / 10 ^ (SNR / 10) / M / N / D;
% n = sqrt(variance) * randn([D M * N]);
% mixed = mixed' + n;
% clear n;

% remove noise
% [UU, SS, VV] = svds(mixed, c);
% Lowmixed = UU' * mixed;
% mixed = UU * Lowmixed;
% EM = UU' * A;


% Print that the program has finished
disp('Finished');
