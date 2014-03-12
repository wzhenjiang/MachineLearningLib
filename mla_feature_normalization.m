function [normalized_data, mu, sigma] = mla_feature_normalization(data, mu = [], sigma = [])
%% Purpose:	This function is to normalize features to a certain range
%% Purpose: The range is [.]
%% Attention: Your application need to save mu and sigma for later normalization of your input X
%% Attention: If mu and sigma is presented, normalization would be according to given value

% initiate variable
normalized_data = [];

% prepare mu and sigma (1 * n matrix)
if size(mu,1) == 0  || size(sigma,1) == 0
	mu = mean(data);
	sigma = std(data);
end;

% form normalized_data
[m,n] = size ( data );

for i = 1: m
	normalized_data = [normalized_data; (data(i,:) - mu ) ./ sigma];
end;

end