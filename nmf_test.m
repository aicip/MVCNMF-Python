clear;
rng(0);  % Set the seed for reproducibility

% --- Parameters --- %
% input parameters
% Syntheitc Data
use_synthetic_data = 1; % 1 for synthetic data, 0 for real data
input_mat_name = 'A.mat';
% bands = (1:4);
bands_mat_name = 'BANDS.mat';
% Landsat Data
% use_synthetic_data = 0; % 1 for synthetic data, 0 for real data
% input_mat_name = 'Landsat_separate_images_BR_R002.mat';
% input_mat_name = 'Landsat.mat';

% mvcnmf parameters
c = 5; % number of endmembers
SNR = 20; %dB
tol = 1e-6;
maxiter = 5;
T = 0.015;
showflag = 0;
verbose = 'on'; % 'on' or 'off'

% --- read data --- %
input_path = sprintf('inputs/%s', input_mat_name);
% Load the list of variables in the .mat file
variables = who('-file', input_path);

% start the timer
tic

% loop through the variables
for i = 1:length(variables)
    clc;
    disp("#########################################")
    fprintf("Processing %s/%s images\n", num2str(i), num2str(length(variables)));
    disp("#########################################")
    variable_name = variables{i};
    %     variable_name = "BR_R002_23KPR00_2014_01_09";
    % Load the first variable in the list
    loaded_variable = load(input_path, variable_name);
    % Set variable A equal to the loaded variable
    A = loaded_variable.(variable_name);

    if use_synthetic_data == 1
        % Load bands
        if ~exist('bands_mat_name', 'var') && ~exist('bands', 'var')
            A = A(:, (1:c));
        else

            disp("using bands file");
            bands_path = sprintf('inputs/%s', bands_mat_name);
            load(bands_path);
            A = A(BANDS, (1:c));

        end

    end

    % --- process --- %

    % print_summary(A, "A");
    if use_synthetic_data == 1
        [synthetic, abf] = getSynData(A, 7, 0);

        % print_summary(synthetic, "synthetic");
        % print_summary(abf, "abf");

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

        % print_summary(UU, "UU")
        % print_summary(Lowmixed, "Lowmixed")

        mixed = UU * Lowmixed;
        EM = UU' * A;
        % vca algorithm
        [A_vca, EndIdx] = vca(mixed, 'Endmembers', c, 'SNR', SNR, 'verbose', verbose);
    else
        % load data
        [M, N, D] = size(A);
        mixed = reshape(A, M * N, D);
        mixed = mixed';
        % create and empty var for UU
        UU = [];
        % vca algorithm
        [A_vca, EndIdx] = vca(mixed, 'Endmembers', c, 'verbose', verbose);
    end

    % print_summary(mixed, "mixed");

    % FCLS
    warning off;
    AA = [1e-5 * A_vca; ones(1, length(A_vca(1, :)))];
    % print_summary(AA, "AA")
    s_fcls = zeros(length(A_vca(1, :)), M * N);
    
    for j = 1:M * N
        r = [1e-5 * mixed(:, j); 1];
        %   s_fcls(:,j) = nnls(AA,r);
        s_fcls(:, j) = lsqnonneg(AA, r);
    end
    % print_summary(s_fcls, "s_fcls")

    % use vca to initiate
    Ainit = A_vca;
    sinit = s_fcls;

    % % random initialization
    % idx = ceil(rand(1,c)*(M*N-1));
    % Ainit = mixed(:,idx);
    % sinit = zeros(c,M*N);

    % PCA
    [PrinComp, pca_score] = pca(mixed');
    % print_summary(PrinComp, "PrinComp")
    meanData = mean(pca_score);
    %[PrinComp, pca_score] = princomp(mixed', 0);
    %meanData = mean(mixed');

    % use conjugate gradient to find A can speed up the learning
    maxiter_str = sprintf('%d', maxiter);
    [Aest, sest] = mvcnmf(mixed, Ainit, sinit, A, UU, PrinComp, meanData, T, tol, maxiter, showflag, 2, 1, use_synthetic_data);

    % visualize endmembers in scatterplots

    if showflag
        d = 4;
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

    if use_synthetic_data == 1
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

    end

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
    if use_synthetic_data == 1
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
    end

    % Save output
    % keep only 2 digits after the decimal point
    T_str = sprintf('%.4f', T);
    outputFileName = sprintf('outputs/output_%s_max_iter%s_T%s.mat', variable_name, maxiter_str, T_str);

    if use_synthetic_data == 1
        save(outputFileName, 'Aest', 'sest', 'E_rmse', 'E_aad', 'E_aid', 'E_sad', 'E_sid');
    else
        save(outputFileName, 'Aest', 'sest');
    end

    break
end

% Stop the timer
elapsed_time = toc;

% Display the elapsed time in seconds
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);
% Program finished
disp('Finished');


function print_summary(array, name)
    disp("---------------------------")
    fprintf('Size of %s: ', name);
    disp(size(array));
    fprintf('Minimum value: %.2f\n', min(array, [], 'all'));
    fprintf('Maximum value: %.2f\n', max(array, [], 'all'));
    fprintf('Mean value: %.2f\n', mean(array, 'all'));
    fprintf('Standard deviation: %.2f\n\n', std(array, 0, 'all'));
    disp("---------------------------")
end