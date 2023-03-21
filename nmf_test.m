clear;

% --- Parameters --- %
% input parameters
use_synthetic_data = 0; % 1 for synthetic data, 0 for real data
% Syntheitc Data
% input_mat_name = 'A.mat';
% bands_mat_name = 'BANDS.mat';
% bands = (1:4);
% Landsat Data
input_mat_name = 'Landsat_separate_images.mat';
c = 18; % number of endmembers

% mvcnmf parameters
SNR = 20; %dB
tol = 1e-6;
maxiter = 150;
T = 0.015;
showflag = 0;
verbose = 'on'; % 'on' or 'off'

% --- read data --- %
input_path = sprintf('inputs/%s', input_mat_name);
% Load the list of variables in the .mat file
variables = who('-file', input_path);
% Load the first variable in the list
loaded_variable = load(input_path, variables{1});
% Set variable A equal to the loaded variable
A = loaded_variable.(variables{1});
if use_synthetic_data == 1
    % Load bands
    if isempty(bands_mat_name) || isempty(bands)
        A = A(:, (1:c));
    else
        if isempty(bands_mat_name)
            disp("using code-specified bands");
            A = A(bands, (1:c));
        else
            disp("using bands file");
            bands_path = sprintf('inputs/%s', bands_mat_name);
            load(bands_path);
            A = A(BANDS, (1:c));
        end
    end
end

% --- process --- %
% start the timer
tic

if use_synthetic_data == 1
    [synthetic, abf] = getSynData(A, 7, 0);
    [M, N, D] = size(synthetic);
    mixed = reshape(synthetic, M * N, D);
    % add noise
    variance = sum(mixed(:) .^ 2) / 10 ^ (SNR / 10) / M / N / D;
    n = sqrt(variance) * randn([D M * N]);
    mixed = mixed' + n;
    clear n;
    
    % remove noise
    [UU, SS, VV] = svds(mixed, c);
    Lowmixed = UU' * mixed;
    mixed = UU * Lowmixed;
    EM = UU' * A;

    % vca algorithm
    [A_vca, EndIdx] = vca(mixed, 'Endmembers', c, 'SNR', SNR, 'verbose', verbose);
else
    % load data
    [M, N, D] = size(A);
    mixed = reshape(A, M * N, D);

    % vca algorithm
    [A_vca, EndIdx] = vca(mixed, 'Endmembers', c, 'verbose', verbose);
end

% FCLS
warning off;
AA = [1e-5 * A_vca; ones(1, length(A_vca(1, :)))];
s_fcls = zeros(length(A_vca(1, :)), M * N);

for j = 1:M * N
    r = [1e-5 * mixed(:, j); 1];
    % print j and r shape
    fprintf('j = %d, r shape = %d\n', j, size(r));
    
    %   s_fcls(:,j) = nnls(AA,r);
    s_fcls(:, j) = lsqnonneg(AA, r);
end

% use vca to initiate
Ainit = A_vca;
sinit = s_fcls;

% % random initialization
% idx = ceil(rand(1,c)*(M*N-1));
% Ainit = mixed(:,idx);
% sinit = zeros(c,M*N);

% PCA
[PrinComp, pca_score] = pca(mixed');
meanData = mean(pca_score);
%[PrinComp, pca_score] = princomp(mixed', 0);
%meanData = mean(mixed');

% use conjugate gradient to find A can speed up the learning
[Aest, sest] = mvcnmf(mixed, Ainit, sinit, A, UU, PrinComp, meanData, T, tol, maxiter, showflag, 2, 1);

% visualize endmembers in scatterplots
d = 4;

if showflag
    Anmf = UU' * Aest;
    figure,

    for i = 1:d - 1

        for j = i + 1:d - 1
            subplot(d - 2, d - 2, (i - 1) * (d - 2) + j - i),
            plot(Lowmixed(i, 1:6:end), Lowmixed(j, 1:6:end), 'rx');
            hold on, plot(EM(i, :), EM(j, :), 'go', 'markerfacecolor', 'g');
            plot(Anmf(i, :), Anmf(j, :), 'bo', 'markerfacecolor', 'b');
        end

    end

end

% permute results
CRD = corrcoef([A Aest]);
DD = abs(CRD(c + 1:2 * c, 1:c));
perm_mtx = zeros(c, c);
aux = zeros(c, 1);

for i = 1:c
    [ld, cd] = find(max(DD(:)) == DD);
    ld = ld(1); cd = cd(1); % in the case of more than one maximum
    perm_mtx(ld, cd) = 1;
    DD(:, cd) = aux; DD(ld, :) = aux';
end

Aest = Aest * perm_mtx;
sest = sest' * perm_mtx;
Sest = reshape(sest, [M, N, c]);
sest = sest';

% show the estimations
if showflag
    figure,

    for i = 1:c
        subplot(c, 4, 4 * i - 3),
        plot(A(:, i), 'r'); axis([0 300 0 1])

        if i == 1
            title('True end-members');
        end

        subplot(c, 4, 4 * i - 2),
        plot(Aest(:, i), 'g'); axis([0 300 0 1])

        if i == 1
            title('Estimated end-members');
        end

        subplot(c, 4, 4 * i - 1),
        imagesc(reshape(abf(i, :), M, N));

        if i == 1
            title('True abundance');
        end

        subplot(c, 4, 4 * i),
        imagesc(Sest(:, :, i));

        if i == 1
            title('Estimated abundance');
        end

    end

end

% quantitative evaluation of spectral signature and abundance

% rmse error of abundances
E_rmse = sqrt(sum(sum(((abf - sest) .* (abf - sest)) .^ 2)) / (M * N * c));
display(E_rmse);

% the angle between abundances
nabf = diag(abf * abf');
nsest = diag(sest * sest');
ang_beta = 180 / pi * acos(diag(abf * sest') ./ sqrt(nabf .* nsest));
E_aad = mean(ang_beta .^ 2) ^ .5;
display(E_aad);

% cross entropy between abundance
E_entropy = sum(abf .* log((abf +1e-9) ./ (sest +1e-9))) + sum(sest .* log((sest +1e-9) ./ (abf +1e-9)));
E_aid = mean(E_entropy .^ 2) ^ .5;
display(E_aid);

% the angle between material signatures
nA = diag(A' * A);
nAest = diag(Aest' * Aest);
ang_theta = 180 / pi * acos(diag(A' * Aest) ./ sqrt(nA .* nAest));
E_sad = mean(ang_theta .^ 2) ^ .5;
display(E_sad);

% the spectral information divergence
pA = A ./ (repmat(sum(A), [length(A(:, 1)) 1]));
qA = Aest ./ (repmat(sum(Aest), [length(A(:, 1)) 1]));
qA = abs(qA);
SID = sum(pA .* log((pA +1e-9) ./ (qA +1e-9))) + sum(qA .* log((qA +1e-9) ./ (pA +1e-9)));
E_sid = mean(SID .^ 2) ^ .5;
display(E_sid);

% Stop the timer
elapsed_time = toc;

% Display the elapsed time in seconds
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);

% Save output
outputFileName = sprintf('outputs/output_%s', input_mat_name);

save(outputFileName, 'Aest', 'sest', 'E_rmse', 'E_aad', 'E_aid', 'E_sad', 'E_sid');
% Program finished
disp('Finished');
