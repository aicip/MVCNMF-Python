function [mixed, abf] = getSynData(A, win, pure)
    %
    % Generate synthetic data.
    % The spectra of simulated data is obtained from the USGS library "signatures"
    %
    % Input
    %   - A: matrix of reflectances
    %   - win: size of smoothing filter
    %   - pure: 0 - no pure pixels, 1 - exist pure pixel
    %
    % Output
    %   - mixed: generated synthetic mixed data
    %   - abf: actual abundance fractions
    %
    % The pure pixels can be removed by adding the following two lines
    %        ----Index = ceil(find(abf>0.8)/c);
    %        ----abf(:,Index) = 1/c*ones(c,1)*ones(1,length(Index));
    %


    [band, c] = size(A);
    dim = 64;

    label = ones((dim/8)^2,1);
    num = floor(length(label)/c);

    for i=1:c-1
        label((i-1)*num+1:i*num) = (i+1); 
    end
            
    ridx = randperm(length(label));
    label = label(ridx)';
    label = reshape(label,dim/8,dim/8);
    abf = zeros(dim,dim,c);
    img = zeros(dim,dim);
    for i=1:dim
        for j=1:dim
            for cls = 1:c
                if label(floor((i-1)/8)+1,floor((j-1)/8)+1) == cls
                    tmp = zeros(c,1);
                    tmp(cls) = 1;
                    abf(i,j,:) = tmp;
                    img(i,j) = c;
                end
            end
        end
    end
    % print_summary(abf, "abf-1")
    
    %low pass filter
    H = ones(win,win)/(win*win);
    % img_fil = filter2(H,img);
    for i=1:c
        abf(:,:,i) = filter2(H,abf(:,:,i));
    end

    abf = abf(ceil(win/2):end-floor(win/2),ceil(win/2):end-floor(win/2),:);    
    
    % generate mixtures
    [M,N,c] = size(abf);

    abf = reshape(abf,M*N,c)';
    
    % remove pure pixels
    if pure == 0
        Index = ceil(find(abf>0.8)/c);
        abf(:,Index) = 1/c*ones(c,1)*ones(1,length(Index));
    end
    % print_summary(A, "A-before")
    % print_summary(abf, "abf-before")
    
    mixed = reshape((A*abf)',M,N,band);
    % print_summary(mixed, "mixed-before")
end 
 
% function print_summary(array, name)
%     disp("---------------------------")
%     fprintf('Size of %s: ', name);
%     disp(size(array));
%     fprintf('Minimum value: %.2f\n', min(array, [], 'all'));
%     fprintf('Maximum value: %.2f\n', max(array, [], 'all'));
%     fprintf('Mean value: %.2f\n', mean(array, 'all'));
%     fprintf('Standard deviation: %.2f\n\n', std(array, 0, 'all'));
%     disp("---------------------------")
% end