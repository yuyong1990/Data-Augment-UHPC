%% === Load Original Dataset ===
load('Database.mat');  % Assumes variable 'data' (n0 x m)

%% === Set Target Number of Samples After Augmentation ===
targetN = 500;

%% === Data Augmentation Using PCA Copula with Boundary Preservation ===
augData = augmentCopulaBounded(data, targetN);

%% === Save Augmented Data ===
save('augmentedData.mat', 'augData');

%% === Compute and Display Statistics for Original and Augmented Data ===
statNames = {'Minimum', 'Maximum', 'Mean', 'Std Dev', 'Median', 'Skewness', 'Kurtosis'};
statsFunc = {
    @(x) min(x);
    @(x) max(x);
    @(x) mean(x);
    @(x) std(x);
    @(x) median(x);
    @(x) skewness(x);
    @(x) kurtosis(x)
};

numStats = numel(statsFunc);
numVars = size(data, 2);

fprintf('\n%-10s | %-12s | %-12s | %-12s\n', 'Variable', 'Statistic', 'Original', 'Augmented');
fprintf('---------------------------------------------------------------\n');

for j = 1:numVars
    for k = 1:numStats
        origVal = statsFunc{k}(data(:,j));
        augVal  = statsFunc{k}(augData(:,j));
        fprintf('Var%-7d | %-12s | %-12.4f | %-12.4f\n', j, statNames{k}, origVal, augVal);
    end
end

%% === Function: PCA Copula based Data Augmentation ===
function augData = augmentCopulaBounded(data, targetN)
% augmentCopulaBounded
% Performs Gaussian copula-based data augmentation in PCA space,
% ensuring new samples preserve dependency structure and boundaries.
%
% Inputs:
%   data    - Original dataset (nSamples x nVariables)
%   targetN - Desired total number of samples after augmentation
%
% Output:
%   augData - Augmented dataset (original + synthetic)

rng(1000);  % For reproducibility

[n0, m] = size(data);
nNew = targetN - n0;

if nNew <= 0
    augData = data;
    return;
end

%% === Step 1: PCA Decomposition ===
[coeff, score, latent, ~, explained, mu] = pca(data);  % score: (n0 x m)

%% === Step 2: Convert PCA scores to Uniform Marginals ===
U_pca = zeros(n0, m);
for j = 1:m
    U_pca(:,j) = tiedrank(score(:,j)) / (n0 + 1);
end

%% === Step 3: Estimate Gaussian Copula in PCA space ===
Rho_pca = corr(norminv(U_pca));

%% === Step 4: Sample from Gaussian Copula ===
mixRatio = 0.5;             % Percentage of auxiliary (more random) samples
nAux = round(nNew * mixRatio);
nCopula = nNew - nAux;

Z_copula = mvnrnd(zeros(1, m), Rho_pca, nCopula);
Z_aux    = mvnrnd(zeros(1, m), Rho_pca, nAux);

U_copula = normcdf(Z_copula);
U_aux    = normcdf(Z_aux);

%% === Step 5: Map back to PCA score space using inverse empirical CDF ===
score_new_copula = zeros(nCopula, m);
score_new_aux    = zeros(nAux, m);

for j = 1:m
    sSorted = unique(sort(score(:,j)));
    cdfVals = linspace(0, 1, length(sSorted));
    
    score_new_copula(:, j) = interp1(cdfVals, sSorted, U_copula(:, j), 'pchip', 'extrap');
    score_new_aux(:, j)    = interp1(cdfVals, sSorted, U_aux(:, j), 'pchip', 'extrap');
end

score_new = [score_new_copula; score_new_aux];

%% === Step 6: Map new PCA scores back to original space ===
Xnew = score_new * coeff' + mu;

%% === Step 7: Match mean/std and clamp to original variable bounds ===
Xorig_bounds = [min(data); max(data)];

for j = 1:m
    mu_orig = mean(data(:, j));
    sigma_orig = std(data(:, j));
    
    mu_new = mean(Xnew(:, j));
    sigma_new = std(Xnew(:, j));
    
    if sigma_new > 0
        Xnew(:, j) = (Xnew(:, j) - mu_new) / sigma_new * sigma_orig + mu_orig;
    else
        Xnew(:, j) = mu_orig;
    end

    % Clamp to original bounds
    Xnew(:, j) = max(min(Xnew(:, j), Xorig_bounds(2,j)), Xorig_bounds(1,j));
end

%% === Final Combine ===
augData = [data; Xnew];

end
